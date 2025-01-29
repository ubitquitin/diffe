from helpers import (
    call_llm,
    call_llm_for_score,
    generate_reasoning_steps,
    extract_key_topics,
    compute_tfidf,
    compute_semantic_similarity,
    compute_topic_connections,
    generate_subquestions,
    calculate_flesch_score,
    calculate_gunning_fog,
)

from termcolor import colored



# Agent 1: Reasoning Steps Counter
def agent_reasoning_steps(question):
    steps = generate_reasoning_steps(question)
    prompt = f"""
    You are an expert at evaluating the difficulty of reasoning steps. Based on the following steps, decide how difficult the question must be to answer.
    Assign a score for all of the steps as a whole, from 0 to 30, where 0 is the easiest and 30 is the hardest:
    
    Steps:
    {steps}
    
    Score:
    """
    response = call_llm_for_score(prompt)
    score = response.score

    print(colored(f"After seeing {len(steps)} reasoning steps, the Reasoning Agent returns a reasoning score of : {score}",'red'))

    return int(score)


# Agent 2 - Topic Relevance
def agent_topic_relevance(question, central_topic):

    # Compute additional metrics
    technical_terms, term_tfidf = compute_tfidf(question, central_topic)
    semantic_similarity = compute_semantic_similarity(question, central_topic)

    # Prepare input for the LLM
    prompt = f"""
    You are an expert at evaluating the difficulty of topic relevance. Based on the following information, assign a score from 0 to 30, where 0 is the easiest and 30 is the hardest:

    Question:
    {question}

    Central Topic:
    {central_topic}

    Technical Terms:
    {technical_terms}

    TF-IDF Scores for Technical Terms:
    {term_tfidf}

    Semantic Similarity to Central Topic:
    {semantic_similarity}

    Consider the following when assigning a score:
    - How niche or specialized the technical terms are.
    - How closely the question aligns with the central topic.
    - The importance of the terms to the central topic (based on TF-IDF scores).

    Score:
    """
    response = call_llm_for_score(prompt)
    score = response.score
    
    print(colored(f"By using central topics: {central_topic}, the Topic Relevance Agent found {len(technical_terms)} technical terms. Using TFIDF and semantic similarity, it returns a topic nuance score of: {score}", 'blue'))
    
    return int(score)


# Agent 3 - Connections/Breadth
def agent_connections(question):
    # Extract key topics
    key_topics = extract_key_topics(question)

    # Compute the number of connections between topics
    num_connections = compute_topic_connections(key_topics)

    # Generate subquestions
    subquestions = generate_subquestions(question)

    # Prepare input for the LLM
    prompt = f"""
    You are an expert at evaluating the difficulty of interdisciplinary connections. Based on the following information, assign a score from 0 to 30, where 0 is the easiest and 30 is the hardest:

    Question:
    {question}

    Key Topics:
    {key_topics}

    Number of Connections Between Topics:
    {num_connections}

    Subquestions:
    {subquestions}

    Consider the following when assigning a score:
    - The breadth of topics covered in the question.
    - The number of connections a learner must make between these topics.
    - The number of subquestions a learner must answer to address the main question.

    Score:
    """
    response = call_llm_for_score(prompt)
    score = response.score
    
    print(colored(f"The Agent finds within the question, the following key topics: {key_topics}\nWith {len(subquestions)} subquestions and {num_connections} connections, the Connections Agent returns a score of {score}.", 'yellow'))
    
    return int(score)


# Agent 4 - Semantic complexity
def agent_semantic_complexity(question):
    # Reading difficulty metrics
    flesch_score = calculate_flesch_score(question)
    gunning_score = calculate_gunning_fog(question)

    # Prepare input for the LLM
    prompt = f"""
    You are an expert at evaluating the semantic difficulty of educational content. Based on the following information, assign a score from 0 to 30, where 0 is the easiest and 30 is the hardest:

    Question:
    {question}

    Flesch Reading Score:
    {flesch_score}

    Gunning FOG Score:
    {gunning_score}

    Score:
    """
    response = call_llm_for_score(prompt)
    score = response.score
    
    print(colored(f"The Agent finds a Flesch Reading score of {flesch_score}, and a Gunning FOG score of {gunning_score}. The Semantic Complexity Agent returns a score of: {score}", 'green'))
    
    return int(score)


# Agent 5: Final Difficulty Scorer
def agent_final_scorer(question, central_topics):
    reasoning_score = agent_reasoning_steps(question)
    topic_score = agent_topic_relevance(question, central_topics)
    connections_score = agent_connections(question)
    complexity_score = agent_semantic_complexity(question)

    # Calculate final score
    final_score = reasoning_score + topic_score
    final_score += connections_score
    final_score += complexity_score
    return final_score


if __name__ == "__main__":
    # Example question
    question = "A triangle has sides measuring 3 cm, 4 cm, and 5 cm. What type of triangle is this?"
    print(colored(f"Question: {question}", 'cyan'))
    # Calculate difficulty score
    difficulty_score = agent_final_scorer(question, central_topics="Geometry")
    print(f"Difficulty Score: {difficulty_score}/100")

    question2 = "A circle is inscribed inside a square with a side length of 10 cm. A smaller square is inscribed inside the circle. What is the area of the smaller square?"
    print(colored(f"Question: {question2}", 'cyan'))
    # Calculate difficulty score
    difficulty_score = agent_final_scorer(question2, central_topics="Geometry")
    print(f"Difficulty Score: {difficulty_score}/100")