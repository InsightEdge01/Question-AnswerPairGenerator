import streamlit as st
import tempfile
import os
import base64
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

prompt_template_questions = """
You are an expert in creating practice questions based on study material.
Your goal is to prepare a student for their exam. You do this by asking questions about the text below:

------------
{text}
------------

Create questions that will prepare the student for their exam. Make sure not to lose any important information.

QUESTIONS:
"""

PROMPT_QUESTIONS = PromptTemplate(template=prompt_template_questions, input_variables=["text"])

refine_template_questions = """
You are an expert in creating practice questions based on study material.
Your goal is to help a student prepare for an exam.
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below.
------------
{text}
------------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.

QUESTIONS:
"""

REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_questions,
)

# Initialize Streamlit app
st.title('Question-Answer Pair Generator with Mixtral 8X7B')
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)

# File upload widget for a single file
uploaded_file = st.sidebar.file_uploader("Upload PDF file", type=["pdf"])

# Set file path
file_path = None

# Check if file is uploaded
if uploaded_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

# Display PDF document
if file_path:
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)

# Check if file_path is set
if file_path:
    question_answer_pairs = []

    # Load data from the uploaded PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # Combine text from Document into one string for question generation
    text_question_gen = ''
    for page in data:
        text_question_gen += page.page_content

    # Initialize Text Splitter for question generation
    text_splitter_question_gen = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=50)

    # Split text into chunks for question generation
    text_chunks_question_gen = text_splitter_question_gen.split_text(text_question_gen)

    # Convert chunks into Documents for question generation
    docs_question_gen = [Document(page_content=t) for t in text_chunks_question_gen]

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
    # Initialize Large Language Model for question generation
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"  
    llm_question_gen = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.2}
    )

    # Initialize question generation chain
    question_gen_chain = load_summarize_chain(llm=llm_question_gen, chain_type="refine", verbose=True,
                                              question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS)
    # Run question generation chain
    questions = question_gen_chain.run(docs_question_gen)

    # Initialize Large Language Model for answer generation
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
    llm_answer_gen = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.2}
    )

    # Create vector database for answer generation
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

    # Initialize vector store for answer generation
    vector_store = Chroma.from_documents(docs_question_gen, embeddings)

    # Initialize retrieval chain for answer generation
    answer_gen_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, chain_type="stuff",
                                                   retriever=vector_store.as_retriever(k=2))

    # Split generated questions into a list of questions
    question_list = questions.split("\n")

    # Answer each question and save to a file
    for question in question_list:
        st.write("Question: ", question)
        answer = answer_gen_chain.run(question)
        question_answer_pairs.append([question, answer])
        st.write("Answer: ", answer)
        st.write("--------------------------------------------------\n\n")

    # Create a directory for storing answers
    answers_dir = os.path.join(tempfile.gettempdir(), "answers")
    os.makedirs(answers_dir, exist_ok=True)

    # Create a DataFrame from the list of question-answer pairs
    qa_df = pd.DataFrame(question_answer_pairs, columns=["Question", "Answer"])

    # Save the DataFrame to a CSV file
    csv_file_path = os.path.join(answers_dir, "questions_and_answers.csv")
    qa_df.to_csv(csv_file_path, index=False)

    # Create a download button for the questions and answers CSV file
    st.markdown('### Download Questions and Answers in CSV')
    st.download_button("Download Questions and Answers (CSV)", csv_file_path)

    # Cleanup temporary file
    os.remove(file_path)
