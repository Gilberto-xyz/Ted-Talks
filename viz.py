# Streamlit webapp
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import streamlit as st


st.title('Ted Talks: Ideas que merecen ser difundidas')
st.write('''
En la vida de todos existe un momento clave en el que entendemos lo que está sucediendo, desde darte cuenta de que también le gustas a esa persona especial, descubrir tu salsa favorita o entender cómo es que funciona while, do while.

También existen momentos en los que te frustras y no puedes seguir con el trabajo. Volver a encontrar el camino suele ser un poco difícil y a veces imposible por la mentalidad de no ser suficiente, de no poder lograrlo, de mejor no intentarlo.

Es aquí el porqué elegí este tema. Siempre vamos a necesitar unos de otros, porque no importa lo lejos que hayas llegado, siempre parece no ser suficiente. 
Gran parte de las charlas disponibles en TED se enfocan en esto, dado que es un gran problema y pasa en todos los sectores.

La curiosidad me llevó a analizar este Dataset, pero en parte fue por lo mucho que frecuentaba algunas charlas que me han ayudado a no rendirme con mi sueño de ser Científico de Datos 


![do it](https://i.ytimg.com/vi/3njZSDjW7Q4/hqdefault.jpg)

_TED es una organización sin ánimo de lucro dedicada a la difusión de ideas, normalmente en forma de charlas breves e impactantes. 
TED comenzó en 1984 como una conferencia en la que convergían la Tecnología, el Entretenimiento y el Diseño, y en la actualidad 
abarca casi todos los temas desde la ciencia hasta los negocios y los problemas globales en más de 110 idiomas._ 
''')

st.header('Sobre el dataset')
st.markdown('''
El dataset está extraído directamente de la página oficial de [Ted Talks](https://www.ted.com/)
por lo que la información es diferente la del canal oficial de [YouTube](https://www.youtube.com/user/TEDxTalks)

- Sería interesante también poder explorar datos extraídos de Youtube para ver la brecha de información de la página oficial a la de YouTube.

[Dataset en Kaggle](https://www.kaggle.com/ashishjangra27/ted-talks)

''')

st.header('EDA - Análisis exploratorio de datos')

# Cargamos el dataset y mostramos los primeros 10 registros

talks_df = pd.read_csv('raw_data/ted_talks.csv')
talks_df.head(10)
# talks_df.tail(10)

# %% [markdown]
# Este dataset contiene registros hasta 2022.

# %%
talks_df.info()

# %% [markdown]
# Existen 5440 pero tenemos un dato faltante en la columna de Autores.

# %%
# Mostramos el registro del autor que falta
talks_df[talks_df['author'].isnull()]

# %% [markdown]
# Esta charla es un resumen, el cual no contiene un autor y se eliminara para la integridad del dataset.
# 
# (Te invito a ver el resumen, es interesante ver como algunas tecnologías evolucionaron rápidamente, pero también algunos problemas se mantuvieron a pesar de los años)
# 

# %%
# Dataframe que excluye el registro de autor faltante
talks_clean = talks_df[~talks_df['author'].isnull()]

# %%
talks_clean.info()

# %% [markdown]
# Mostramos los datos únicos de cada columna dentro del dataset.

# %%
for columna in talks_clean.columns:
    print(f'Valores unicos en la columna {columna}: {talks_clean[columna].nunique()}')

# %% [markdown]
# - Los Valores únicos en la columna author son: 4,443. Lo que indica que existen autores con más de una charla en el dataset.
# 
# - Los Valores únicos en la columna title son: 5,439. (Es el número total de registros con los que estamos trabajando)
# 

# %%
# Restamos el numero total de registros y el numero de autores unicos para obtener el numero de autores faltantes
talks_clean['author'].count() - talks_clean['author'].nunique() 

# %%
# Mostrar las celdas que contienen autores repetidos
talks_clean[talks_clean['author'].duplicated()]

# Contamos todos los registros que contienen autores repetidos
# talks_clean[talks_clean['author'].duplicated()].count()

# %% [markdown]
# La fecha contiene un formato poco habitual, por lo que se dividirá en mes y año.

# %%
talks_clean[['month', 'year']] = talks_clean['date'].str.split(' ', expand=True)

# %%
# Eliminar la columna de date
talks_clean.drop(columns=['date'], inplace=True)

# %%
# Cual es la primer charla en los registros de Ted Talks?

# Ordenamos el dataset por fecha y mostramos el primer registro
talks_clean.sort_values(by=['year'], ascending=True).head(1)

# %%
# Numero de platicas por año. De mayor a menor
talks_clean['year'].value_counts(sort=True)

# %%
# Cuantos años hay en el dataset?
talks_clean['year'].nunique()

# %%
# Top 10 ted talks con mas likes
talks_clean.groupby('title')['likes'].sum().sort_values(ascending = False).head(10)


# %%
# Top 10 ted talks con mas vistas
talks_clean.groupby('title')['views'].sum().sort_values(ascending = False).head(10)







# Data Viz
###################################################################################################################
# Dataset
talks_df = pd.read_csv('clean_data/ted_talks_clean.csv')
talks_df.head()

coment_words = ''
stopwords = set(STOPWORDS)

for val in talks_df.title:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    coment_words += ' '.join(tokens)+' '

wordcloud = WordCloud(width = 2000, height = 1200,
                background_color ='white',
                stopwords = stopwords,
                colormap='Set2',
                max_words = 100,
                min_font_size = 0).generate(coment_words)

plt.figure(figsize = (20,20), facecolor = None)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# %% [markdown]
# # Top 10 Ted Talks por cantidad de likes
# 
# Los likes son una medida interna de la plataforma que permite valorar una charla si el usuario está registrado 
# 
# Graficado con:
# - Matplotlib

# %%
# variables para graficar
titulo = talks_df.groupby('title')['likes'].sum().sort_values(ascending = False).head(10).index
likes = talks_df.groupby('title')['likes'].sum().sort_values(ascending = False).head(10)

# formato numeros con coma 
def formato_numeros(valor, index):
    if valor == 0:
        formato = '{:1.1f} '.format(valor)
    elif valor >= 1_000_000:
        formato = '{:1.1f} M'.format(valor * 0.000_001)
    else:
        formato = '{:1.0f} K'.format(valor * 0.001)
    return formato


# %%
# Estilo
plt.style.use('ggplot')

# Tamaño del grafico
fig, ax = plt.subplots(figsize=(25,10))

# Tipo de grafico
ax.barh(titulo, likes)

# Espacio  de barras
step_value = likes.max()/20

# Formato numeros
ax.xaxis.set_major_formatter(formato_numeros)

# Titulo
ax.set_title('Las 10 Charlas con más likes', fontsize=15)
# Fondo blanco
ax.set_facecolor('white')

# Etiqueta al final de cada barra
for i, v in enumerate(likes):
    ax.text(v+step_value/5, i, formato_numeros(v, i), color='gray')

plt.show()


# %% [markdown]
# # Top 10 Ted Talks por visitas
# Las vistas son consideradas según la frecuencia del recurso, estas no requieren inscribirse y pueden incrementar gracias a un mismo usuario
# 
# Graficado con:
# - Matplotlib

# %%
# Variables para graficar
title = talks_df.groupby('title')['views'].sum().sort_values(ascending = False).head(10).index
views = talks_df.groupby('title')['views'].sum().sort_values(ascending = False).head(10)

# %%
# Grafica
plt.style.use('ggplot')

# Tamaño de la figura
fig, ax = plt.subplots(figsize=(25,10))

# Grafica
ax.barh(title, views)

# Formato numeros con coma
ax.xaxis.set_major_formatter(formato_numeros)

# Titulo
ax.set_title('Las 10 Charlas con más visitas', fontsize=15)
# Fondo blanco
ax.set_facecolor('white')

step_value=views.max()/20
# Etiquetas al final de cada barra
for i, v in enumerate(views):
    ax.text(v+step_value/5, i, formato_numeros(v, i), color='gray')

plt.show()

# %% [markdown]
# 

# %% [markdown]
# # Top 10 TED Talks que tienen la mejor relación entre likes y visualizaciones
# 
# Graficado con :
# - Matplotlib

# %%
# Variables para graficar
# Nueva columna con la relacion de likes/views
talks_df['like_view_ratio'] = (talks_df['likes']/talks_df['views'])*100
ratio = talks_df[['title', 'like_view_ratio']].head(10).sort_values(by='like_view_ratio', ascending=False)
# Redondeo de los valores de ratio en 2 decimales 
ratio['like_view_ratio'] = ratio['like_view_ratio'].round(2)

# Tamaño de la figura
fig, ax = plt.subplots(figsize=(20,10))

# Grafica
ax.barh(ratio.title, ratio.like_view_ratio)

# Titulo
ax.set_title('Las 10 Charlas con mejor relación entre likes y visualizaciones', fontsize=15)
# Fondo blanco
ax.set_facecolor('white')

step_value=ratio.like_view_ratio.max()/20


# Etiquetas al final de cada barra
for i, v in enumerate(ratio.like_view_ratio):
    ax.text(v+step_value/5, i, v, color='gray')

plt.show()

# %% [markdown]
# # Charlas por año
# 
# Nos ayuda a obtener una "primera vista" general, o panorama de las charlas por año.
# 
# Graficado con:
# - Matplotlib

# %%
# Variable para graficar

total = talks_df['year'].value_counts().sort_index(ascending=True)

plt.style.use('ggplot')

# Tamaño de la figura
fig, ax = plt.subplots(figsize=(30,10))

# Puntos de la grafica
ax.scatter(total.index, total.values, s=total.values, alpha=0.3)
# Linea de la grafica
ax.plot(total.index, total.values)

# Etiquetas X, Y para valores en el grafico
for i, txt in enumerate(total.values):
    ax.annotate(txt, (total.index[i] + .4, total.values[i]),color='gray')

# Fondo blanco
ax.set_facecolor('white')
# Titulo
ax.set_title('Numero de charlas por año de publicación', fontsize=15)

# Años de publicación en X
plt.xticks(ticks=total.index, labels=total.index, rotation=45)

plt.show()

# %%
# Variable para graficar

# Mostrar solo los años de publicación a partir de 2001
total = total[total.index > 2000]

# Tamaño de la figura
fig, ax = plt.subplots(figsize=(30,10))

# Puntos de la grafica
ax.scatter(total.index, total.values, s=total.values, alpha=0.3)
# Linea de la grafica
ax.plot(total.index, total.values)

# Etiquetas X, Y para valores en el grafico
for i, txt in enumerate(total.values):
    ax.annotate(txt, (total.index[i] + .4, total.values[i]),color='gray', ha = 'center', va='center')

# Fondo blanco
ax.set_facecolor('white')
# Titulo
ax.set_title('Charlas por año de publicación: 2001 - 2022', fontsize=15)

# Años de publicación en X
plt.xticks(ticks=total.index, labels=total.index, rotation=45)

plt.show()

# %% [markdown]
# # Los 10 autores con más charlas
# 
# Graficado con: 
# - Matplotlib

# %%
# Variables para graficar
speaker = talks_df.author.value_counts().head(10)

# Grafico
fig, ax = plt.subplots(figsize=(29,10))

ax.barh(speaker.index, speaker.values)


# Titulo
ax.set_title('Los 10 autores con más charlas', fontsize=15)
# Fondo blanco
ax.set_facecolor('white')

step_value=speaker.max()/20
# Etiquetas al final de cada barra
for i, v in enumerate(speaker):
    ax.text(v+step_value/5, i, v, color='gray')

plt.show()


# %% [markdown]
# # Los 10 Autores más populares por número de visitas
# 
# Gráficado con:
# - Matplotlib

# %%
# Variables para graficar
authot_views = talks_df.groupby('author')['views'].mean().nlargest(10).sort_values(ascending=False)

# Grafico
fig, ax = plt.subplots(figsize=(30,10))

ax.barh(authot_views.index, authot_views.values)

# Titulo
ax.set_title('Los 10 autores con más visualizaciones', fontsize=15)
# Fondo blanco
ax.set_facecolor('white')

step_value=authot_views.max()/20

# Formato numeros
ax.xaxis.set_major_formatter(formato_numeros)

# Etiquetas al final de cada barra
for i, v in enumerate(authot_views):
    ax.text(v+step_value/5, i, formato_numeros(v, i), color='gray')

plt.show()

# %% [markdown]
# # Número de charlas en 2019
# 
# Graficado con:
# - Matplotlib

# %%
# Datos del 2019
talks_df_2019 = talks_df[talks_df['year'] == 2019]
# Vairble para ordenar los meses
order = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December')
# Guardamos en una nueva columna la lista de meses en el orden correcto
talks_df_2019['order'] = pd.Categorical(talks_df_2019['month'], categories=order, ordered=True)
# Ordenamos por el orden de la lista y actualizamos el dataframe
talks_df_2019.sort_values(by=['order'], inplace=True)

# %%
# Variable para graficar 
mes2k19 = talks_df_2019.groupby('order')['title'].count()

# Grafico 
fig, ax = plt.subplots(figsize=(30,15))

ax.bar(mes2k19.index, mes2k19.values)

# Titulo 
ax.set_title('2019 - Charlas por Mes', fontsize=15)
# Fondo blanco
ax.set_facecolor('white')

step_value=mes2k19.max()/20

# Etiquetas al final de cada barra vertical 
for i, v in enumerate(mes2k19):
    ax.text(i, v+step_value/5, v, color='gray')

plt.show()

# %% [markdown]
# # Visitas por Año
# 
# Graficado con: 
# - Matplotlib

# %%
# Variables para graficar
years_views = talks_df.groupby('year')['views'].sum()

# Tamaño de la figura
fig, ax = plt.subplots(figsize=(30,15))

# Linea de la grafica
ax.bar(years_views.index, years_views.values)
step_value=years_views.max()/20

# Fondo blanco
ax.set_facecolor('white')
# Titulo
ax.set_title('Visualizaciones por año', fontsize=15)

# Años de publicación en X
plt.xticks(ticks=years_views.index, labels=years_views.index, rotation=45)

# Etiquetas en X, Y para valores en el grafico con formato numeros y step_value para que quede alto
for i, v in enumerate(years_views):
    ax.annotate(formato_numeros(v, i), (years_views.index[i], years_views.values[i]),color='gray', ha='center', va='bottom', fontsize=9)

plt.show()

# %%

# Mostrar desde el 2001 hasta la actualidad
years_views = years_views[years_views.index >= 2001]

# Tamaño de la figura
fig, ax = plt.subplots(figsize=(30,15))

# Linea de la grafica
ax.bar(years_views.index, years_views.values)
step_value=years_views.max()/20

# Fondo blanco
ax.set_facecolor('white')
# Titulo
ax.set_title('Visualizaciones por año', fontsize=15)

# Años de publicación en X
plt.xticks(ticks=years_views.index, labels=years_views.index, rotation=45)

# Etiquetas en X, Y para valores en el grafico con formato numeros y step_value para que quede alto
for i, v in enumerate(years_views):
    ax.annotate(formato_numeros(v, i), (years_views.index[i], years_views.values[i]),color='gray', ha='center', va='bottom', fontsize=9)

plt.show()
