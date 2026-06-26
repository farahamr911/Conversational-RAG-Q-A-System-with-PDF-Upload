
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.memory.motorhead_memory import MotorheadMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

class pipeline:
    
    def __init__(self,db,llm):
        self.db=db
        self.llm=llm
        self.history_store={}

    def memory_init(self,session_id:str):
        memory=MotorheadMemory(
            url="http://localhost:8000",
            session_id=session_id,
            memory_key="hist.",
            return_messages=True  
        )
        return memory
    
    def add_user_message(self, message:BaseMessage,session_id:str):
        memory=self.memory_init(session_id=session_id)
        memory.chat_memory.add_user_message(message=message)
        self.history_store.setdefault(session_id,[]).append(HumanMessage(content=message.content))


    def add_AI_message(self,message:BaseMessage,session_id):
        memory=self.memory_init(session_id=session_id)
        memory.chat_memory.add_ai_message(message=message)
        self.history_store.setdefault(session_id,[]).append(AIMessage(content=message.content))    

    

    def build_rag_pipeline(self,session_id:str):
        retriever = self.db.as_retriever(search_kwargs={"k": 4})
        memory = self.memory_init(session_id)
        #document_chain=create_stuff_documents_chain(llm,prompt)
        """def combine_docs(docs):
            return "\n\n".join(
            getattr(d, "page_content", str(d)) for d in docs
            )"""
        def combine_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        
        combine_docs_chain = RunnableLambda(combine_docs)

        def get_memory_history(memory):
           memory_vars = memory.load_memory_variables({})
           return memory_vars.get("history", [])
        
        #history
        history_chain = RunnableLambda(lambda _: get_memory_history(memory))

        #prompt
        prompt=ChatPromptTemplate.from_template(
        """
    Answer the questions based on the provided context and history.
    Please provide the most accurate response based on the question
context
{context}

question
{question}

history
{history}
"""
          )
    

        retrieval_chain = (
        { 
            "question": RunnableLambda(lambda x: x["question"]),   # Extract question
            "context": (
                RunnableLambda(lambda x: x["question"])            # Extract question
                | retriever
                | combine_docs_chain
            ),
            "history":history_chain
            #"context": retriever | combine_docs_chain,   
            #"question": RunnablePassthrough()                                                 
        }
        | prompt
        | self.llm
        )
        return retrieval_chain


