#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head(10)


# In[5]:


black_friday['Age'].value_counts()


# In[6]:


black_friday.dtypes


# In[7]:


black_friday.query("Age == '26-35' & Gender == 'F'").shape[0]


# In[8]:


black_friday.loc[(black_friday["Gender"] == 'F') & (black_friday["Age"] == '26-35')].shape[0]


# In[9]:


black_friday_by_Age_and_Gender = black_friday.groupby(['Age','Gender']).agg({'User_ID':'count'}).reset_index()
black_friday_by_Age_and_Gender['User_ID'][4]


# In[10]:


black_friday['User_ID'].nunique()


# In[11]:


black_friday.dtypes.nunique()


# In[12]:


(black_friday.isna().sum()/black_friday.shape[0]).max()


# In[13]:


black_friday.isna().sum().max()


# In[14]:


black_friday['Product_Category_3'].dropna().mode()


# In[15]:


normal = (black_friday['Purchase'] - black_friday['Purchase'].min()) / (black_friday['Purchase'].max() - black_friday['Purchase'].min())
normal.mean()


# In[16]:


padron = (black_friday['Purchase'] - black_friday['Purchase'].mean()) / black_friday['Purchase'].std()
padron[padron.between(-1,1)].shape[0]


# In[17]:


bool((black_friday[black_friday['Product_Category_2'].isna()][['Product_Category_2','Product_Category_3']]).isnull().values.all())


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[28]:


def q1():
    return black_friday.shape
    pass
q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[5]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return black_friday.query("Age == '26-35' & Gender == 'F'").shape[0]


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[6]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday['User_ID'].nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[7]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[8]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return (black_friday.isna().sum()/black_friday.shape[0]).max()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[9]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return black_friday.isna().sum().max()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[10]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return float(black_friday['Product_Category_3'].dropna().mode())


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[14]:


def q8():
    # Retorne aqui o resultado da questão 8.
    normal = ((black_friday['Purchase'] - black_friday['Purchase'].min()) / 
              (black_friday['Purchase'].max() - black_friday['Purchase'].min()))
    return normal.mean()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[15]:


def q9():
    # Retorne aqui o resultado da questão 9.
    padron = (black_friday['Purchase'] - black_friday['Purchase'].mean()) / black_friday['Purchase'].std()
    return padron[padron.between(-1,1)].shape[0]


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[17]:


def q10():
    # Retorne aqui o resultado da questão 10.
    var_null = black_friday[black_friday['Product_Category_2'].isna()][['Product_Category_2','Product_Category_3']]
    return bool(var_null.isnull().values.all())

