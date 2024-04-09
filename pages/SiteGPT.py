import streamlit as st
import os
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler


# https://openai.com/sitemap.xml

answers_prompt = ChatPromptTemplate.from_template(
    """
    답변할 수 없다면 아무말이나 지어내지말고 모른다고 하세요.
    그리고 각 답변을 0부터 5까지의 점수로 평가해주세요.
    0점은 사용자에게 쓸모없음, 5점은 사용자에게 매우 유용함을 의미합니다.
    사용자 question에 대한 예제입니다.
    
    Make sure to inclide the answer's score.
    
    Context: {context}
    
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!
    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.
            
            반드시 점수가 높고 유저에게 도움이 되는 질문을 하나만 골라서 답해주세요.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(input):
    answers = input["answers"]
    question = input["question"]
    choose_chain = choose_prompt | llm
    condence = "\n\n".join(
        f"Answer: {answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}\n"
        for answer in answers
    )
    print(choose_chain.invoke({"question": question, "answers": condence}))
    return choose_chain.invoke({"question": question, "answers": condence})


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " '")


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/blog\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 5  # 속도는 느려지지만 차단은 안 당한다
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


st.set_page_config(
    page_title="Site GPT",
    page_icon="🖥️",
)

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.

"""
)

with st.sidebar:
    openaikey = None
    openaikey = st.text_input("Your OpenAI API key: ", type="password")
    os.environ["OPENAI_API_KEY"] = openaikey

if openaikey:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
    )

    with st.sidebar:
        url = st.text_input("Write down a URL", placeholder="https://example.com")

    if url:

        if ".xml" not in url:
            with st.sidebar:
                st.error("Please write down a url")
        else:
            retriever = load_website(url)
            query = st.text_input("Ask a question to the website.")
            if query:
                chain = (
                    {"docs": retriever, "question": RunnablePassthrough()}
                    | RunnableLambda(get_answers)
                    | RunnableLambda(choose_answer)
                )
                result = chain.invoke(query)
                st.write(result.content.replace("$", "\$"))
