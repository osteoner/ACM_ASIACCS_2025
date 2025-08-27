import json
import logging
import os
import random
import textwrap
from collections import Counter, defaultdict
import cycler,re

from regex import T

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from umap import UMAP
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from cycler import cycler
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

default_cycler = (cycler(color=[
    "#0F0A0A",
    "#FF6600",
    "#FFBE00",
    "#496767",
    "#87A19E",
    "#FF9200",
    "#0F3538",
    "#F8E08E",
    "#0F2021",
    "#FAFAF0"])
    )
plt.rc('axes', prop_cycle=default_cycler)

DEFAULT_INSTRUCTION = (
    instruction
) = """You are an expert in underground forum and community analysis.\
Given a short description of a community’s dominant topics, generate a concise and descriptive name (3–5 words) that clearly summarizes its focus.\  
Avoid overly generic terms and ensure the name reflects the main activity or theme. \
Only return the name, no explanations.\

Example:\
Description: discussions about Bitcoin trading, mining rigs, and altcoin speculation\
Output: Cryptocurrency & Finance\
"""

DEFAULT_TEMPLATE = "<s>[INST]{examples}\n\n{instruction}[/INST]"


class ClusterClassifier:
    def __init__(
        self,
        embed_model_name = "google/flan-t5-base",
        embed_device="cpu",
        embed_batch_size=64,
        embed_max_seq_length=512,
        embed_agg_strategy=None,
        umap_components=2,
        umap_metric="cosine",
        dbscan_eps=0.09,
        dbscan_min_samples=10,
        dbscan_n_jobs=16,
        summary_create=False,
        summary_model = "Qwen/Qwen2.5-7B-Instruct", #change to qwen3
        topic_mode="multiple_topics",
        summary_n_examples=10,
        summary_chunk_size=420,
        summary_model_token=False,
        summary_template=None,
        summary_instruction=None,
    ):
        self.embed_model_name = embed_model_name
        self.embed_device = embed_device
        self.embed_batch_size = embed_batch_size
        self.embed_max_seq_length = embed_max_seq_length
        self.embed_agg_strategy = embed_agg_strategy

        self.umap_components = umap_components
        self.umap_metric = umap_metric

        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_n_jobs = dbscan_n_jobs

        self.summary_create = summary_create
        self.summary_model = summary_model
        self.topic_mode = topic_mode
        self.summary_n_examples = summary_n_examples
        self.summary_chunk_size = summary_chunk_size
        self.summary_model_token = summary_model_token

        if summary_template is None:
            self.summary_template = DEFAULT_TEMPLATE
        else:
            self.summary_template = summary_template

        if summary_instruction is None:
            self.summary_instruction = DEFAULT_INSTRUCTION
        else:
            self.summary_instruction = summary_instruction

        self.embeddings = None
        self.faiss_index = None
        self.cluster_labels = None
        self.texts = None
        self.projections = None
        self.umap_mapper = None
        self.id2label = None
        self.label2docs = None

        self.embed_model = SentenceTransformer(
            self.embed_model_name, device=self.embed_device
        )
        self.embed_model.max_seq_length = self.embed_max_seq_length
        
        # Add special tokens to the embedding model
        special_tokens = ["[M]", "[T]", "[R]", "[SEP]"]
        tokenizer = self.embed_model.tokenizer
        
        # Add tokens to tokenizer
        tokenizer.add_tokens(special_tokens)
        
        # Resize model embeddings to accommodate new tokens
        for module in self.embed_model.modules():
            if hasattr(module, 'auto_model'):
                module.auto_model.resize_token_embeddings(len(tokenizer))
                break

    def fit(self, texts, embeddings=None):
        self.texts = texts

        if embeddings is None:
            logging.info("embedding texts...")
            self.embeddings = self.embed(texts)
        else:
            logging.info("using precomputed embeddings...")
            self.embeddings = embeddings

        logging.info("building faiss index...")
        self.faiss_index = self.build_faiss_index(self.embeddings)
        logging.info("projecting with umap...")
        self.projections, self.umap_mapper = self.project(self.embeddings)
        logging.info("dbscan clustering...")
        self.cluster_labels = self.cluster(self.projections)

        self.id2cluster = {
            index: label for index, label in enumerate(self.cluster_labels)
        }
        self.label2docs = defaultdict(list)
        for i, label in enumerate(self.cluster_labels):
            self.label2docs[label].append(i)

        self.cluster_centers = {}
        for label in self.label2docs.keys():
            x = np.mean([self.projections[doc, 0] for doc in self.label2docs[label]])
            y = np.mean([self.projections[doc, 1] for doc in self.label2docs[label]])
            self.cluster_centers[label] = (x, y)

        if self.summary_create:
            logging.info("summarizing cluster centers...")
            self.cluster_summaries = self.summarize(self.texts, self.cluster_labels)
        else:
            self.cluster_summaries = None
        #save embeddings and cluster labels
        np.save("embeddings.npy", self.embeddings)
        np.save("cluster_labels.npy", self.cluster_labels)
        return self.embeddings, self.cluster_labels, self.cluster_summaries

    def infer(self, texts, top_k=1):
        embeddings = self.embed(texts)

        dist, neighbours = self.faiss_index.search(embeddings, top_k)
        inferred_labels = []
        for i in tqdm(range(embeddings.shape[0])):
            labels = [self.cluster_labels[doc] for doc in neighbours[i]]
            inferred_labels.append(Counter(labels).most_common(1)[0][0])

        return inferred_labels, embeddings

    def embed(self, texts):
        embeddings = self.embed_model.encode(
            texts,
            batch_size=self.embed_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings

    def project(self, embeddings):
        mapper = UMAP(n_components=self.umap_components, metric=self.umap_metric).fit(
            embeddings
        )
        return mapper.embedding_, mapper

    def cluster(self, embeddings):
        print(
            f"Using DBSCAN (eps, nim_samples)=({self.dbscan_eps,}, {self.dbscan_min_samples})"
        )
        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            n_jobs=self.dbscan_n_jobs,
        ).fit(embeddings)

        return clustering.labels_

    def build_faiss_index(self, embeddings):
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def summarize(self, texts, labels):
        unique_labels = len(set(labels)) - 1  # exclude the "-1" label
        
        # Initialize local pipeline instead of API client
        tokenizer = AutoTokenizer.from_pretrained(self.summary_model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.summary_model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        
        cluster_summaries = {-1: "None"}

        for label in range(unique_labels):
            ids = np.random.choice(self.label2docs[label], self.summary_n_examples)
            examples = "\n\n".join(
                        [
                            "{}".format(
                                re.sub(r'\[[^\]]*\]', '', texts[_id].split('[T]')[1][:self.summary_chunk_size])
                            )
                            for i, _id in enumerate(ids)
                        ]
                    )
                                    
            # print(examples)
            request = self.summary_template.format(
                examples=examples, instruction=self.summary_instruction
            )
            
            # Use local pipeline instead of API call
            response = pipeline(request, max_new_tokens=200, do_sample=True, temperature=0.7)
            response_text = response[0]['generated_text']
            
            # Extract only the generated part (remove the input prompt)
            if request in response_text:
                response_text = response_text.replace(request, "").strip()
            
            if label == 0:
                print(f"Request:\n{request}")
            cluster_summaries[label] = self._postprocess_response(response_text)
        print(f"Number of clusters is {len(cluster_summaries)}")
        return cluster_summaries

    def _postprocess_response(self, response):
        if self.topic_mode == "multiple_topics":
            summary = response.split("\n")[0].split(".")[0].split("(")[0]
            summary = ",".join(
                [txt for txt in summary.strip().split(",") if len(txt) > 0]
            )
            return summary
        elif self.topic_mode == "single_topic":
            first_line = response.split("\n")[0]
            topic, score = None, None
            try:
                topic = first_line.split("Topic:")[1].split("(")[0].split(",")[0].strip()
            except IndexError:
                print("No topic found")
            try:
                score = first_line.split("Educational value rating:")[1].strip().split(".")[0].strip()
            except IndexError:
                print("No educational score found")
            full_output = f"{topic}. Educational score: {score}"
            return full_output
        else:
            raise ValueError(
                f"Topic labeling mode {self.topic_mode} is not supported, use single_topic or multiple_topics instead."
            )

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(f"{folder}/embeddings.npy", "wb") as f:
            np.save(f, self.embeddings)

        faiss.write_index(self.faiss_index, f"{folder}/faiss.index")

        with open(f"{folder}/projections.npy", "wb") as f:
            np.save(f, self.projections)

        with open(f"{folder}/cluster_labels.npy", "wb") as f:
            np.save(f, self.cluster_labels)

        # Convert pandas Series to list if needed
        texts_to_save = self.texts
        if hasattr(self.texts, 'tolist'):  # Check if it's a pandas Series/DataFrame
            texts_to_save = self.texts.tolist()
        elif not isinstance(self.texts, list):
            texts_to_save = list(self.texts)

        with open(f"{folder}/texts.json", "w") as f:
            json.dump(texts_to_save, f)

        with open(f"{folder}/mistral_prompt.txt", "w") as f:
            f.write(DEFAULT_INSTRUCTION)

        if self.cluster_summaries is not None:
            with open(f"{folder}/cluster_summaries.json", "w") as f:
                json.dump(self.cluster_summaries, f)

    def load(self, folder):
        if not os.path.exists(folder):
            raise ValueError(f"The folder '{folder}' does not exsit.")

        with open(f"{folder}/embeddings.npy", "rb") as f:
            self.embeddings = np.load(f)

        self.faiss_index = faiss.read_index(f"{folder}/faiss.index")

        with open(f"{folder}/projections.npy", "rb") as f:
            self.projections = np.load(f)

        with open(f"{folder}/cluster_labels.npy", "rb") as f:
            self.cluster_labels = np.load(f)

        with open(f"{folder}/texts.json", "r") as f:
            self.texts = json.load(f)

        if os.path.exists(f"{folder}/cluster_summaries.json"):
            with open(f"{folder}/cluster_summaries.json", "r") as f:
                self.cluster_summaries = json.load(f)
                keys = list(self.cluster_summaries.keys())
                for key in keys:
                    self.cluster_summaries[int(key)] = self.cluster_summaries.pop(key)

        # those objects can be inferred and don't need to be saved/loaded
        self.id2cluster = {
            index: label for index, label in enumerate(self.cluster_labels)
        }
        self.label2docs = defaultdict(list)
        for i, label in enumerate(self.cluster_labels):
            self.label2docs[label].append(i)

        self.cluster_centers = {}
        for label in self.label2docs.keys():
            x = np.mean([self.projections[doc, 0] for doc in self.label2docs[label]])
            y = np.mean([self.projections[doc, 1] for doc in self.label2docs[label]])
            self.cluster_centers[label] = (x, y)

    def show(self, interactive=False):
        df = pd.DataFrame(
            data={
                "X": self.projections[:, 0],
                "Y": self.projections[:, 1],
                "labels": self.cluster_labels,
                "content_display": [
                    textwrap.fill(txt[:1024], 64) for txt in self.texts
                ],
            }
        )

        if interactive:
            self._show_plotly(df)
        else:
            self._show_mpl(df)

    def _show_mpl(self, df):
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        df["color"] = df["labels"].apply(lambda x: "C0" if x==-1 else f"C{(x%9)+1}")

        df.plot(
            kind="scatter",
            x="X",
            y="Y",
            # c="labels",
            s=0.75,
            alpha=0.8,
            linewidth=0,
            color=df["color"],
            ax=ax,
            colorbar=False,
        )

        for label in self.cluster_summaries.keys():
            if label == -1:
                continue
            summary = self.cluster_summaries[label]
            position = self.cluster_centers[label]
            t= ax.text(
                position[0],
                position[1],
                summary,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=4,
            )
            t.set_bbox(dict(facecolor='white', alpha=0.9, linewidth=0, boxstyle='square,pad=0.1'))
        ax.set_axis_off()

    def _show_plotly(self, df):
        fig = px.scatter(
            df,
            x="X",
            y="Y",
            color="labels",
            hover_data={"content_display": True, "X": False, "Y": False},
            width=1600,
            height=800,
            color_continuous_scale="HSV",
        )

        fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")

        fig.update_traces(
            marker=dict(size=1, opacity=0.8),  # color="white"
            selector=dict(mode="markers"),
        )

        fig.update_layout(
            template="plotly_dark",
        )

        # show cluster summaries
        for label in self.cluster_summaries.keys():
            if label == -1:
                continue
            summary = self.cluster_summaries[label]
            position = self.cluster_centers[label]

            fig.add_annotation(
                x=position[0],
                y=position[1],
                text=summary,
                showarrow=False,
                yshift=0,
            )

        fig.show()