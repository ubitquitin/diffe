import textstat
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel

load_dotenv()


class LLMScore(BaseModel):
    score: int


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def call_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"{prompt}"},
        ],
        stream=False
    )

    return response.choices[0].message.content.strip()


def call_llm_for_score(prompt):
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"{prompt}"},
        ],
        response_format=LLMScore
    )

    return response.choices[0].message.parsed


def generate_reasoning_steps(question):
    prompt = f"Break down the minimal number reasoning steps to answer the following question. If the question is simple information recall, this should just be one step. If the question requires many possible subquestions, break it down into several steps. Enclose each step in <step> tags:\n\n{question}"
    response = call_llm(prompt)  # Replace with actual API call
    import re

    steps = re.findall(r"<step>(.*?)</step>", response)
    return steps


# Function to compute TF-IDF for technical terms
def compute_tfidf(question, central_topic):

    # Load a pre-trained NLP model
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(question)

    # Extract technical terms
    technical_terms = [
        token.text for token in doc if token.pos_ in ["NOUN", "ADJ"] and token.is_alpha
    ]

    # Compute TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([central_topic, question])
    tfidf_scores = dict(
        zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[1])
    )

    # Get TF-IDF scores for technical terms
    term_tfidf = {term: tfidf_scores.get(term, 0) for term in technical_terms}
    return technical_terms, term_tfidf


# Function to compute semantic similarity to the central topic
def compute_semantic_similarity(question, central_topic):

    # Load a pre-trained model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode the question and central topic
    question_embedding = model.encode(question)
    topic_embedding = model.encode(central_topic)

    # Compute cosine similarity
    similarity = util.cos_sim(question_embedding, topic_embedding).item()
    return similarity


# Function to extract key topics from the question
def extract_key_topics(question):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(question)
    key_topics = [
        token.text
        for token in doc
        if token.pos_ in ["NOUN", "PROPN"] and token.is_alpha
    ]
    key_topics = list(set(key_topics))
    return key_topics


# Function to compute the number of connections between topics
def compute_topic_connections(key_topics):

    # Load a pre-trained model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode the key topics
    topic_embeddings = model.encode(key_topics)

    # Compute pairwise cosine similarities
    similarity_matrix = util.cos_sim(topic_embeddings, topic_embeddings)

    # Count the number of significant connections (e.g., similarity > 0.5)
    num_connections = sum(
        similarity_matrix[i][j] > 0.5
        for i in range(len(key_topics))
        for j in range(i + 1, len(key_topics))
    )
    return num_connections


# Function to generate subquestions
def generate_subquestions(question):
    prompt = f"""
    Break down the following question into subquestions that a learner might need to answer to fully address the main question. Enclose each subquestion in <subquestion> tags:

    Question:
    {question}

    Subquestions:
    """
    response = call_llm(prompt)  # Replace with actual API call
    import re

    subquestions = re.findall(r"<subquestion>(.*?)</subquestion>", response)
    return subquestions


def calculate_flesch_score(question):
    return textstat.flesch_reading_ease(question)


def calculate_gunning_fog(question):
    return textstat.gunning_fog(question)
