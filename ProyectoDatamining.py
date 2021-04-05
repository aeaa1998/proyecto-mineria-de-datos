#!/usr/bin/env python
# coding: utf-8

# # Proyecto 1 Data mining
# ## Realizado por:
# ### Nombres 
# - Augusto Alonso - 181085
# - Joohno Molina -
# - Mario Sarmientos -

# In[1]:


get_ipython().system('pip install plotly')


# In[2]:


get_ipython().system('pip install pyreadstat')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install xlrd==1.2.0')
get_ipython().system('pip install sklearn')


# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyreadstat
import numpy as np
import plotly.express as px


# In[4]:


a = range(2009,2020)
b = "Data/defunciones"
lista = []
nombres = ['Depreg', 'Mupreg', 'Mesreg', 'Añoreg', 'Depocu', 'Mupocu', 'Areag',
       'Sexo', 'Diaocu', 'Mesocu', 'Añoocu', 'Edadif', 'Perdif', 'Getdif',
       'Ecidif', 'Ocudif', 'Dnadif', 'Mnadif', 'Nacdif', 'Dredif', 'Mredif',
       'Caudef', 'Asist', 'Ocur', 'Cerdef', 'year']
palabras = ['mupreg', 'mupocu','añoocu', 'Escodif', 'mnadif i in columnas:if', 'Pnadif', 'Predif', 'Puedif', 'Ciuodif',
       'caudef.descrip']


# In[5]:


for k in a:
    path = b + str(k) + ".sav"
    df, meta = pyreadstat.read_sav(path)
    columnas = df.columns
    columnas = list(columnas) 
    for i in columnas:
        if i in palabras:
            for j in nombres:
                
                if i.lower() == j.lower():
                    df = df.rename(columns={i: j})
    df["year"] = k
    lista.append(df)


# In[6]:


defunciones = pd.concat(lista)
defunciones.shape
defunciones.columns


# In[7]:


defunciones.head()


# In[8]:


defunciones = defunciones[defunciones.columns[:-6]]
defunciones.head()


# ### Observamos que tenemos algunas columnas con data perdida

# In[9]:


defunciones.isna().any()


# Se decidio llenar los vacios con 0 Puesto que si la columna es categórica entonces no habría problema
# porque a la hora de contar no se categorizaría al elemento

# In[10]:


data = defunciones.fillna(0)
data.head()


# In[11]:


data.info()


# In[12]:


data.describe()


# In[13]:


dic = pd.read_excel("Data/diccionario.xlsx",sheet_name="Defunciones",skiprows = [0])
dic["Valor"].unique()


# Notese que los departamentos que tienen más fallecidos son Guatemala, Quetzaltenango, Altaverapaz y San Marcos

# In[14]:


a = data["Depocu"].value_counts()
a[:5]


# In[15]:


sns.countplot(y=data["Depocu"])


# Por otro lado los municipios donde más gente fallece es Guatemala,Quetzaltenango, Escuintla y Mazatenango

# In[16]:


a = data["Mupocu"].value_counts()
a[:5]


# In[17]:


b = a[:5]
sns.barplot(b.index,b)


# Los meses donde la gente fallecio mas fuel mes de Enero, Julio y Mayo. Aunque no hay mucha diferencia con el resto. Yo no la consideraría una variable para un modelo puesto que apenas hay alguna diferencia.

# In[18]:


a = data["Mesocu"].value_counts()
a[:5]


# In[19]:


sns.barplot(a.index,a)


# Si ignoramos la categoría donde se llenaron los datos no disponibles vemos que ha ido en aumento el número de defunciones en Guatemala, en este periodo han aumentado más de 10000 muertes

# In[20]:


a = data["Añoocu"].value_counts()
a


# In[21]:


sns.barplot(a.index,a)


# Claramente vemos como es que los hombres murieron más que las mujeres, sin embargo la proporción es bastante similar

# In[22]:


a = data["Sexo"].value_counts()
a


# In[23]:


plt.pie(a,labels=a.index,autopct='%1.2f%%')
plt.title("Hombres y mujeres fallecidos entre 2009 a 2019")


# Vemos que ha medida que las personas van aumentando de edad también aumenta la probabilidad de fallecer puesto que el histogram tiene una asimetría hacia la izquierda. Sin embargo hasta la izquierda tenemos un gran número de defunciones que son los niños menores de un año 

# In[24]:


sns.catplot(x = "Edadif",
           data = data,
           kind = "box", 
           sym = "", # to ommit the points that are so far
           )
data.Edadif.quantile([0.25,0.5,0.75])


# In[25]:


from matplotlib import rcParams
rcParams['figure.figsize'] = 7.7,4.27
# Los datos atipicos que removimos aca fueron porque hubo un mal ingreso de datos porque el dato era de 999
data = data[data["Edadif"] < 200]
data.hist('Edadif',bins=14)


# Vemos que las personas que no son del grupo índigena fallecieron más que las personas que si pertenecen a este grupo, pero esta variable no la tomaria en cuenta debido a la poca cantidad de datos 

# In[26]:


a = data["Getdif"].value_counts()
a


# In[27]:


sns.barplot(a.index,a)


# Vemos que las personas que están solteras fueron las más fallecieron a lo largo de este período 

# In[28]:


a = data['Ecidif'].value_counts()
a


# In[29]:


sns.barplot(a.index,a)


# Las ocupaciones de las personas que más fallecieron fueron las de trabajos domésticos no remunerados, peones de explotaciones agrícolas, agricultores y trabajadores calificados de cultivos extensivos, Estudiante

# In[30]:


a = data['Ocudif'].value_counts()
a[1:6]


# In[31]:


b = a[1:6]
sns.barplot(b.index,b)


# Los departamentos donde nacieron la mayoria de los difuntos son Guatemala, San Marcos, Alta Verapaz y Huehuetenango

# In[32]:


a = data['Dnadif'].value_counts()
a[:9]


# In[33]:


g = sns.barplot(a.index,a)


# Las causas principales de fallecimiento fueron: Infarto agudo del miocardio, sin otra especificación, Neumonía, no especificada, Diabetes mellitus no especificada, sin mención de complicación, Muerte sin asistencia, Exposición a factores no especificados, causando otras lesiones y las no especificadas.

# In[34]:


a = data['Caudef'].value_counts()
a[:5]


# In[35]:


b = a[:5]
g = sns.barplot(b.index,b)


# La gran mayoría no recibio ninguna asistencia, aunque su proporción es bastante parecida con los que recibieron atención médica, muy por atrás encontramos la asistencia empírica y los otros tipos de asistencia.

# In[36]:


a = data['Asist'].value_counts()
a


# In[37]:


g = sns.barplot(a.index,a)


# In[38]:


data['Asist']


# Podemos observar que a lo largo de los años la cantidad de defunciones ha ido en aumento, podemos observar que en este periodo de 10 años han aumentado las defunciones aproximadamente 

# In[39]:


sns.countplot(y=data["year"])


# ### Cruzando variables

# En la siguiente tabla cruzamos el departamento donde se registro el difunto y el departamento de residia el difunto, encontramos por ejemplo que son más de 50000 personas las que vivían en el departamento de Guatemala pero sin embargo fallecieron fuera de este departamento

# In[40]:


pd_crosstab = pd.crosstab(data["Depreg"], data["Dredif"], margins=True)
pd_crosstab


# In[41]:


# Creamos una columna clasificando a los distintos grupos de edad en categorias
criteria = [data['Edadif'].between(0, 1),data['Edadif'].between(1.1, 10),data['Edadif'].between(10.1, 20),data['Edadif'].between(21.1, 35),
            data['Edadif'].between(35.1, 50),data['Edadif'].between(51.1, 65),data['Edadif'].between(65.1, 100)]
values = ["Menos de un año", "Niño", "Adolescente","Joven","Adulto","Adulto 2","Edad avanzada"]

data['Edadrange'] = np.select(criteria, values, 0)


# Ahora cruzamos los grupos de edad y el sexo donde vemos que las mujeres de edad avanzada fueron las personas que más fallecieron en este período de tiempo seguidas de los hombres.

# In[42]:


pd_crosstab = pd.crosstab(data["Edadrange"], data["Sexo"], margins=True)
pd_crosstab


# In[43]:


sns.heatmap(pd_crosstab,annot=True)


# Los hombres solteros fueron las personas que más fallecieron en este período seguidos de las mujeres solteras

# In[44]:


pd_crosstab = pd.crosstab(data["Ecidif"], data["Sexo"], margins=True)
pd_crosstab


# In[45]:


sns.heatmap(pd_crosstab,annot=True)


# En la siguiente tabla comparamos la causa de defunción con el departamento donde sucede y vemos que en Guatemala es donde más gente fallece por ataques al corazón, seguido de Quetzaltenango y Alta Verapaz.
# 
# Con la Neumonía tenemos a Alta Verapaz como el departamento donde más gente muere por esto, esto puede ser debido a sus temperaturas frías y sus altas temperaturas, seguido de Guatemala y Totonicapán.

# In[46]:


pd_crosstab = pd.crosstab(data["Caudef"], data["Dredif"], margins=True)
pd_crosstab.sort_values("All",ascending=False).head(6)


# A continuación hacemos una tabla del rango de edad y la causa de muerte, donde vemos que las personas mayores fallecieron a causa de enfermedades del corazón, seguida de la diabetes y muerte sin asistencia.
# 
# El segundo grupo más afectado son los adultos que se encuentran entre los 50 y 65 años de edad

# In[47]:


pd_crosstab = pd.crosstab(data["Caudef"], data["Edadrange"], margins=True)
pd_crosstab.sort_values("All",ascending=False).head(6)


# In[48]:


a = pd_crosstab.sort_values("All",ascending=False).head(6)
sns.heatmap(a)


# # Aplicando Kmeans

# In[49]:


from sklearn.cluster import KMeans
data = data.select_dtypes(exclude=['object'])


# In[50]:


lista2 = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, max_iter = 300)
    kmeans.fit(data)
    lista2.append(kmeans.inertia_)


# Al hacer la gráfica de codo vemos que el número optimo de clusters para este conjunto de datos es de 4 puesto que es el que menor error cuadrado tiene y práctiamente tiene la misma precisión con más clusters

# In[51]:


plt.plot(range(1,11),lista2)
plt.title("Gráfica de codo")
plt.xlabel("Número de clusters")
plt.ylabel("Within-Cluster-Sum-of-Squares")


# In[52]:


# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters = 4)

# Fit model to points
model.fit(data)


# Creado ya el modelo creamos una nueva columna para ver a donde pertenece cada individuo

# In[53]:


data['Clusters'] = model.labels_
sns.countplot(y=data["Clusters"])


# Note que el sexo no es un factor tan importante en los primeros 3 clusters, solamente en el cluster 3 donde la mayoria son hombres

# In[54]:


pd_crosstab = pd.crosstab(data["Clusters"], data["Sexo"], margins=True,normalize='index')
pd_crosstab


# En todos los clusters se mantiene casi la misma proporcion en cuanto al departamento donde fallecieron menos en el cluster 3

# In[55]:


pd_crosstab = pd.crosstab(data["Clusters"], data["Depocu"], margins=True,normalize='index')
pd_crosstab


# Notemos que los que pertenecen al cluster 2 y 1 tienen una mayor proporcion de personas con edad avanzada que el cluster 0 

# In[56]:


# Creamos una columna clasificando a los distintos grupos de edad en categorias
criteria = [data['Edadif'].between(0, 1),data['Edadif'].between(1.1, 10),data['Edadif'].between(10.1, 20),data['Edadif'].between(21.1, 35),
            data['Edadif'].between(35.1, 50),data['Edadif'].between(51.1, 65),data['Edadif'].between(65.1, 100)]
values = ["Menos de un año", "Niño", "Adolescente","Joven","Adulto","Adulto 2","Edad avanzada"]

data['Edadrange'] = np.select(criteria, values, 0)

pd_crosstab = pd.crosstab(data["Clusters"], data["Edadrange"], margins=True,normalize='index')
pd_crosstab


# In[59]:


data2 = data.select_dtypes(exclude=['object'])
X = data2.iloc[:,:].values
from sklearn.metrics import silhouette_score
grupos = [KMeans(n_clusters = i, max_iter = 300).fit(X) for i in range (1,11)]


# In[ ]:


scores = [silhouette_score(X, model.labels_) for model in grupos[1:]]
scores


# In[ ]:




