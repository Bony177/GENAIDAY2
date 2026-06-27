

!pip install -q gradio groq langchain langchain-community langchain-groq langchain-huggingface langchain-text-splitters faiss-cpu sentence-transformers pypdf
     

import os
from google.colab import userdata
os.environ["GROQ_API_KEY"]= userdata.get("GROQ_API_KEY")
print("Key is set")

     

import gradio as gr
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
retriever = None

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

prompt = ChatPromptTemplate.from_template("""
You are a website question-answering assistant.

Rules:
1. Answer ONLY from the provided context.
2. Do NOT use outside knowledge.
3. Do NOT guess or make assumptions.
4. If the answer is not found in the context, reply exactly:

I cannot find that information in the website.

Context:
{context}

Question:
{question}

Answer:
""")

grounding_prompt = ChatPromptTemplate.from_template("""
Context:
{context}

Answer:
{answer}

Is every statement in the answer supported by the context?

Reply ONLY:
GROUNDED

or

NOT_GROUNDED
""")

grounding_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

grounding_chain = (
    grounding_prompt
    | grounding_llm
    | StrOutputParser()
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_website(url):
    global retriever

    try:
        loader = WebBaseLoader(url)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        return f"Website loaded successfully. Indexed {len(chunks)} chunks."

    except Exception as e:
        return f"Error loading website: {str(e)}"

def answer_question(question, temperature):
    global retriever

    if retriever is None:
        return "Please load a website first.", ""

    question_lower = question.lower()

    blocked_patterns = [
        "ignore previous instructions",
        "ignore all instructions",
        "reveal your system prompt",
        "system prompt",
        "developer prompt",
        "developer message",
        "jailbreak",
        "act as",
        "pretend to be",
        "bypass",
        "override instructions",
        "forget previous instructions"
    ]

    if any(pattern in question_lower for pattern in blocked_patterns):
        return "This request is not allowed.", ""

    sensitive_keywords = [
        "password",
        "secret",
        "api key",
        "token",
        "private information",
        "confidential information",
        "credit card",
        "bank account"
    ]

    if any(keyword in question_lower for keyword in sensitive_keywords):
        return "Sensitive or confidential information cannot be disclosed.", ""

    docs = retriever.invoke(question)
    print("\nRetrieved Docs:", len(docs))

    for i, doc in enumerate(docs):
      print(f"\n===== CHUNK {i+1} =====")
    print(doc.page_content[:500])

    if not docs:
        return "I cannot find that information in the website.", ""

    context = format_docs(docs)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature
    )

    qa_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = qa_chain.invoke({
        "context": context,
        "question": question
    })

    grounded = (
        grounding_chain.invoke({
            "context": context,
            "answer": answer
        })
        .strip()
        .upper()
    )

    if grounded != "GROUNDED":
        return "I cannot find that information in the website.", ""

    source_chunks = "\n\n".join(
        [
            f"Source Chunk {i+1}:\n{doc.page_content[:500]}"
            for i, doc in enumerate(docs)
        ]
    )

    return answer, source_chunks

with gr.Blocks(title="Website RAG Q&A Bot") as demo:

    gr.Markdown("# Website RAG Q&A Bot")

    website_url = gr.Textbox(
        label="Website URL",
        placeholder="https://example.com"
    )

    load_button = gr.Button("Load Website")

    status = gr.Textbox(
        label="Status",
        interactive=False
    )

    load_button.click(
        load_website,
        inputs=website_url,
        outputs=status
    )

    question = gr.Textbox(
        label="Question",
        placeholder="Ask a question about the website..."
    )

    temperature = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=0.0,
        step=0.1,
        label="Temperature"
    )

    ask_button = gr.Button("Ask")

    answer = gr.Textbox(
        label="Answer",
        lines=8
    )

    sources = gr.Textbox(
        label="Source Chunks",
        lines=12
    )

    ask_button.click(
        answer_question,
        inputs=[question, temperature],
        outputs=[answer, sources]
    )

demo.launch(debug=True)
     
