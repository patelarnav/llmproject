#!/usr/bin/env python
# coding: utf-8

# #### Google Palm LLM & API key setup

# In[117]:


from langchain.llms import GooglePalm

api_key = 'AIzaSyDlXYnP2xNNa0DNa7dPN89u2L4IuAchEg4'

llm = GooglePalm(google_api_key=api_key, temperature=0.2)


# #### Connect with database and ask some basic questions

# In[118]:


from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain


# In[119]:


db_user = "root"
db_password = "root"
db_host = "localhost"
db_name = "atliq_tshirts"

db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info=3)

print(db.table_info)


# In[140]:


db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
qns1 = db_chain("How many t-shirts do we have left for nike in extra small size and white color?")


# Above is the correct answer 👍🏼

# In[122]:


qns2 = db_chain.run("How much is the price of the inventory for all small size t-shirts?")


# It made a mistake here. The price is actually the price per unit but in real life database columns will not have perfect names. We need to tell it somehow that price is price per unit and the actual query should be,
# 
# SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'

# In[123]:


qns2 = db_chain.run("SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'")


# we will use qns2 value later on in this notebook. So hold on for now and let's check another query

# In[124]:


qns3 = db_chain.run("If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue our store will generate (post discounts)?")


# Above, it returned a wrong query which generated an error during query execution. It thinks discount
# table would have start and end date which is normally true but in our table there is no start or end date column.
# One thing we can do here is run the query directly.

# In[ ]:


sql_code = """
select sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
 """

qns3 = db_chain.run(sql_code)


# It produced a correct answer when explicit query was given. 17462 is the total revenue without discounts. The total discount is 736.6. Hence revenue post discount is 17462-736.6=16725.4
# 
# Now this is not much interesting because what is the point of giving it the ready made query? Well, we will use this same query later on for few shot learning

# In[ ]:


qns4 = db_chain.run("SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'")


# In[ ]:


qns5 = db_chain.run("How many white color Levi's t shirts we have available?")


# Once again above is the wrong answer. We need to use SUM(stock_quantity). Let's run the query explicitly. We will use the result of this query later on in the notebook

# In[ ]:


qns5 = db_chain.run("SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'")


# #### Few shot learning
# 
# We will use few shot learning to fix issues we have seen so far

# In[ ]:


few_shots = [
    {'Question' : "How many t-shirts do we have left for Nike in XS size and white color?",
     'SQLQuery' : "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'XS'",
     'SQLResult': "Result of the SQL query",
     'Answer' : qns1},
    {'Question': "How much is the total price of the inventory for all S-size t-shirts?",
     'SQLQuery':"SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'",
     'SQLResult': "Result of the SQL query",
     'Answer': qns2},
    {'Question': "If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue  our store will generate (post discounts)?" ,
     'SQLQuery' : """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
 """,
     'SQLResult': "Result of the SQL query",
     'Answer': qns3} ,
     {'Question' : "If we have to sell all the Levi’s T-shirts today. How much revenue our store will generate without discount?" ,
      'SQLQuery': "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'",
      'SQLResult': "Result of the SQL query",
      'Answer' : qns4},
    {'Question': "How many white color Levi's shirt I have?",
     'SQLQuery' : "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'",
     'SQLResult': "Result of the SQL query",
     'Answer' : qns5
     }
]


# ### Creating Semantic Similarity Based example selector
# 
# - create embedding on the few_shots
# - Store the embeddings in Chroma DB
# - Retrieve the the top most Semantically close example from the vector store

# In[ ]:


from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

to_vectorize = [" ".join(example.values()) for example in few_shots]


# In[ ]:


to_vectorize


# In[ ]:


vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)


# In[ ]:


example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

example_selector.select_examples({"Question": "How many Adidas T shirts I have left in my store?"})


# In[ ]:


### my sql based instruction prompt
mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CURDATE() function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: Query to run with no pre-amble
SQLResult: Result of the SQLQuery
Answer: Final answer here

No pre-amble.
"""


# In[ ]:


from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt

print(PROMPT_SUFFIX)


# ### Setting up PromptTemplete using input variables

# In[ ]:


from langchain.prompts.prompt import PromptTemplate

example_prompt = PromptTemplate(
    input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
    template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
)


# In[ ]:


print(_mysql_prompt)


# In[ ]:


few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=mysql_prompt,
    suffix=PROMPT_SUFFIX,
    input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
)


# In[ ]:


new_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)


# In[ ]:


new_chain("How many white color Levi's shirt I have?")


# Now this is working ok. Previously for this same question it was giving wrong answer because it did not use SUM clause around stock_quantity column

# In[ ]:


new_chain("How much is the price of the inventory for all small size t-shirts?")


# In[ ]:


new_chain("How much is the price of all white color levi t shirts?")


# In[ ]:


new_chain("If we have to sell all the Nike’s T-shirts today with discounts applied. How much revenue  our store will generate (post discounts)?")


# In[ ]:


new_chain("If we have to sell all the Van Heuson T-shirts today with discounts applied. How much revenue  our store will generate (post discounts)?")


# In[ ]:


new_chain.run('How much revenue  our store will generate by selling all Van Heuson TShirts without discount?')

