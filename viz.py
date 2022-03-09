# Streamlit webapp
from importlib import resources
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import streamlit as st

######  
st.title('Ted Talks: Ideas que merecen ser difundidas')
st.write('''
En la vida de todos existe un momento clave en el que entendemos lo que está sucediendo, desde darte cuenta de que también le gustas a esa persona especial, descubrir tu salsa favorita o entender cómo es que funciona while, do while.

También existen momentos en los que te frustras y no puedes seguir con el trabajo. Volver a encontrar el camino suele ser un poco difícil y a veces imposible por la mentalidad de no ser suficiente, de no poder lograrlo, de mejor no intentarlo.

Es aquí el porqué elegí este tema. Siempre vamos a necesitar unos de otros, porque no importa lo lejos que hayas llegado, siempre parece no ser suficiente. 
Gran parte de las charlas disponibles en TED se enfocan en esto, dado que es un gran problema y pasa en todos los sectores.

La curiosidad me llevó a analizar este Dataset, pero en parte fue por lo mucho que frecuentaba algunas charlas que me han ayudado a no rendirme con mi sueño de ser Científico de Datos 
''')
st.image('https://i.redd.it/apqur6fk9t761.jpg', width=700)
st.write('''
_TED es una organización sin ánimo de lucro dedicada a la difusión de ideas, normalmente en forma de charlas breves e impactantes. 
TED comenzó en 1984 como una conferencia en la que convergían la Tecnología, el Entretenimiento y el Diseño, y en la actualidad 
abarca casi todos los temas desde la ciencia hasta los negocios y los problemas globales en más de 110 idiomas._ 
''')
# Info dataset
st.subheader('Sobre el dataset')
st.markdown('''
[Dataset en Kaggle](https://www.kaggle.com/ashishjangra27/ted-talks)

El dataset está extraído directamente de la página oficial de [Ted Talks](https://www.ted.com/)
por lo que la información es diferente la del canal oficial de [YouTube](https://www.youtube.com/user/TEDxTalks)

![ted](https://cdn.iconscout.com/icon/free/png-256/ted-5-282539.png) 
![youtube](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Logo_of_YouTube_%282015-2017%29.svg/502px-Logo_of_YouTube_%282015-2017%29.svg.png)

- Sería interesante también poder explorar datos extraídos de Youtube para ver la brecha de información de la página oficial a la de YouTube.

> **Nota:**
> Los datos y el código fuente de esta aplicacion web se encuentran en el respositorio de GitHub 
> [https://github.com/GilbertoNavaMarcos/Ted-Talks](https://github.com/GilbertoNavaMarcos/Ted-Talks)


''')

##### EDA
st.header('EDA - Análisis exploratorio de datos')
st.subheader('Cargando el dataset')
st.markdown('''
Los datos originales se encuentran en un archivo CSV ubicados en `raw_data/ted_talks.csv`

Hacemos uso de los datos mediante la librería Pandas, con las líneas de código:

```python
import pandas as pd

talks_df = pd.read_csv('raw_data/ted_talks.csv')
```
Mostramos los primeros registros del dataset
```python
talks_df.head()
```
''')
# Load data
talks_df = pd.read_csv('raw_data/ted_talks.csv')
# Mostrar los primeros registros
st.dataframe(talks_df.head())

st.markdown('''
Imprimimos el resumen del Dataframe con la función `.info()`

```python
talks_df.info()
```
''')
import io
buffer = io.StringIO()
talks_df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.markdown('''
### Autor Faltante

> Existen 5440 valores en cada columna pero tenemos un dato faltante en Autores.

Mostramos el registro del autor que falta
```python
talks_df[talks_df['author'].isnull()]
```
''')
st.dataframe(talks_df[talks_df['author'].isnull()])

st.markdown('''
Esta charla es un resumen, el cual no contiene un autor y se eliminara para la integridad del dataset

(Te invito a ver el resumen, es interesante ver como algunas tecnologías evolucionaron rápidamente, pero también algunos problemas se mantuvieron a pesar de los años)

[Year in ideas 2015](https://ted.com/talks/year_in_ideas_2015)

Para excluir esta charla, invertimos la condición de `author` en `isnull()` con `~` y lo guardamos en un nuevo Dataframe

```python
talks_clean = talks_df[~talks_df['author'].isnull()]
```
''')
# Dataframe que excluye el registro de autor faltante
talks_clean = talks_df[~talks_df['author'].isnull()]

st.markdown('''
Mostramos los datos únicos de cada columna dentro del dataset
```python
for columna in talks_clean.columns:
    print(f'Valores unicos en la columna {columna}: {talks_clean[columna].nunique()}')
```
Valores unicos en la columna title: 5439

Valores unicos en la columna author: 4443
> Existen autores con más de una charla en el dataset!

Valores unicos en la columna date: 200

Valores unicos en la columna views: 972

Valores unicos en la columna likes: 752

Valores unicos en la columna link: 5439
''')

st.subheader('Formato de fecha')
st.markdown('''
La fecha contiene un formato poco habitual, por lo que separamos en mes y año.

```python 
talks_clean[['month', 'year']] = talks_clean['date'].str.split(' ', expand=True)
```
Borramos la columa `date` ya que no nos sirve más

```python
talks_clean.drop('date', axis=1, inplace=True)
```
''')
# Separar mes y año
talks_clean[['month', 'year']] = talks_clean['date'].str.split(' ', expand=True)
# Eliminar la columna de date
talks_clean.drop(columns=['date'], inplace=True)

st.subheader('Primer acercamiento a los datos a graficar')
st.markdown('''
Numero de platicas por año. De mayor a menor

```python 
talks_clean['year'].value_counts(sort=True)
```
''')
st.dataframe(talks_clean['year'].value_counts(sort=True))

st.markdown(''' 
Top 10 ted talks con mas vistas
```python 
talks_clean.groupby('title')['views'].sum().sort_values(ascending = False).head(10)
```
''')
st.dataframe(talks_clean.groupby('title')['views'].sum().sort_values(ascending = False).head(10))
st.markdown('''
Exportamos el dataset a un archivo CSV
```python
talks_clean.to_csv('clean_data/ted_talks_clean.csv', index=False)
```
''')

##### Data Viz
st.header('Visualización de datos')
st.subheader('Cargando el dataset limpio')
st.markdown('''
El archivo CSV limpio se encuentra ubicado en `clean_data/ted_talks_clean.csv`

Hacemos uso de los datos mediante la librería Pandas, con las líneas de código:

```python
import pandas as pd

talks_df = pd.read_csv('clean_data/ted_talks_clean.csv')
```
''')
# Dataset
talks_df = pd.read_csv('clean_data/ted_talks_clean.csv')

st.subheader('Nube de palabras')
st.markdown('''
Como primer acercamiento, se trata de plantear que es lo que nos viene a la mente cuando escuchamos Ted Talks. Se genera una nube de palabras con los títulos de las charlas. Cuanto más grande se muestra una palabra, más veces fue empleada.

Codigo:
```python
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

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
```
''')
st.image('https://raw.githubusercontent.com/GilbertoNavaMarcos/Ted-Talks/03956ee8a0c84ea8d4ac9b76687b09a22d928261/resources/ted_talks_wordcloud.svg', width=700)

st.subheader('Las 10 Charlas con más likes')
st.markdown('''
Los likes son una medida interna de la plataforma que permite valorar una charla si el usuario está registrado

codigo:
```python
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
```
''')
titulo = talks_df.groupby('title')['likes'].sum().sort_values(ascending = False).head(10).index
likes = talks_df.groupby('title')['likes'].sum().sort_values(ascending = False).head(10)
def formato_numeros(valor, index):
    if valor == 0:
        formato = '{:1.1f} '.format(valor)
    elif valor >= 1_000_000:
        formato = '{:1.1f} M'.format(valor * 0.000_001)
    else:
        formato = '{:1.0f} K'.format(valor * 0.001)
    return formato
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(25,10))
ax.barh(titulo, likes)
step_value = likes.max()/20
ax.xaxis.set_major_formatter(formato_numeros)
ax.set_title('Las 10 Charlas con más likes', fontsize=15)
ax.set_facecolor('white')
for i, v in enumerate(likes):
    ax.text(v+step_value/5, i, formato_numeros(v, i), color='gray')
# Imprime el grafico en streamlit
st.pyplot(fig)

st.subheader('Las 10 Charlas con más visitas')
st.markdown('''
Las vistas son consideradas según la frecuencia del recurso, estas no requieren inscribirse y pueden incrementar gracias a un mismo usuario.

Codigo:
```python
# Variables para graficar
title = talks_df.groupby('title')['views'].sum().sort_values(ascending = False).head(10).index
views = talks_df.groupby('title')['views'].sum().sort_values(ascending = False).head(10)

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
```
''')
title = talks_df.groupby('title')['views'].sum().sort_values(ascending = False).head(10).index
views = talks_df.groupby('title')['views'].sum().sort_values(ascending = False).head(10)
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(25,10))
ax.barh(title, views)
ax.xaxis.set_major_formatter(formato_numeros)
ax.set_title('Las 10 Charlas con más visitas', fontsize=15)
ax.set_facecolor('white')
step_value=views.max()/20
for i, v in enumerate(views):
    ax.text(v+step_value/5, i, formato_numeros(v, i), color='gray')
# Imprime el grafico en streamlit
st.pyplot(fig)

st.subheader('Top 10 TED Talks que tienen la mejor relación entre likes y visualizaciones')
st.markdown('''
En este gráfico se resalta la relación entre la parte orgánica (vistas) y  las interacciones (likes) para los títulos. Nos aproxima al compromiso por cierto público con los títulos

```python 
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
```
''')
talks_df['like_view_ratio'] = (talks_df['likes']/talks_df['views'])*100
ratio = talks_df[['title', 'like_view_ratio']].head(10).sort_values(by='like_view_ratio', ascending=False)
ratio['like_view_ratio'] = ratio['like_view_ratio'].round(2)
fig, ax = plt.subplots(figsize=(20,10))
ax.barh(ratio.title, ratio.like_view_ratio)
ax.set_title('Las 10 Charlas con mejor relación entre likes y visualizaciones', fontsize=15)
ax.set_facecolor('white')
step_value=ratio.like_view_ratio.max()/20
for i, v in enumerate(ratio.like_view_ratio):
    ax.text(v+step_value/5, i, v, color='gray')
st.pyplot(fig)

st.subheader('Charlas por año')
st.markdown('''
Nos ayuda a obtener una "primera vista", o panorama de las charlas por año.

codigo:
```python 
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
```
''')
total = talks_df['year'].value_counts().sort_index(ascending=True)
fig, ax = plt.subplots(figsize=(30,10))
ax.scatter(total.index, total.values, s=total.values, alpha=0.3)
ax.plot(total.index, total.values)
for i, txt in enumerate(total.values):
    ax.annotate(txt, (total.index[i] + .4, total.values[i]),color='gray')
ax.set_facecolor('white')
ax.set_title('Numero de charlas por año de publicación', fontsize=15)
plt.xticks(ticks=total.index, labels=total.index, rotation=45)
# Imprime el grafico en streamlit
st.pyplot(fig)

st.markdown('''
Zoom - Charlas a partir del año 2000

Codigo:
```python 
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
ax.set_title('Charlas por año de publicación: 2000 - 2022', fontsize=15)

# Años de publicación en X
plt.xticks(ticks=total.index, labels=total.index, rotation=45)

plt.show()
```
''')
total = total[total.index > 2000]
fig, ax = plt.subplots(figsize=(30,10))
ax.scatter(total.index, total.values, s=total.values, alpha=0.3)
ax.plot(total.index, total.values)
for i, txt in enumerate(total.values):
    ax.annotate(txt, (total.index[i] + .4, total.values[i]),color='gray', ha = 'center', va='center')
ax.set_facecolor('white')
ax.set_title('Charlas por año de publicación: 2000 - 2022', fontsize=15)
plt.xticks(ticks=total.index, labels=total.index, rotation=45)
# Imprime el grafico en streamlit
st.pyplot(fig)

st.subheader('Los 10 autores con más charlas')
st.markdown('''
Como se dedujo en el proceso de exploración existen autores con más de una charla en el dataset. Aca se muestran el nombre de los autores con más charlas y el numero de charlas que ha publicado.

Codigo:
```python
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
```
''')
speaker = talks_df.author.value_counts().head(10)
fig, ax = plt.subplots(figsize=(29,10))
ax.barh(speaker.index, speaker.values)
ax.set_title('Los 10 autores con más charlas', fontsize=15)
ax.set_facecolor('white')
step_value=speaker.max()/20
for i, v in enumerate(speaker):
    ax.text(v+step_value/5, i, v, color='gray')
# Imprime el grafico en streamlit
st.pyplot(fig)

st.subheader('Autores más vistos')
st.markdown('''
En algunas ocasiones tener experiencia no asegura el éxito. Esto se comprueba con los siguientes autores que tienen charlas que atraen más al público.

Codigo:
```python
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
```
''')
authot_views = talks_df.groupby('author')['views'].mean().nlargest(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(30,10))
ax.barh(authot_views.index, authot_views.values)
ax.set_title('Los 10 autores con más visualizaciones', fontsize=15)
ax.set_facecolor('white')
step_value=authot_views.max()/20
ax.xaxis.set_major_formatter(formato_numeros)
for i, v in enumerate(authot_views):
    ax.text(v+step_value/5, i, formato_numeros(v, i), color='gray')
# Imprime el grafico en streamlit
st.pyplot(fig)

st.subheader('2019 - Charlas por mes')
st.markdown('''
Volviendo un poco atrás, en el gráfico de “charlas por año de publicación: 2000 - 2022”, se puede notar un récord. En 2019 se han publicado más charlas que en otros años. Aquí se muestran las charlas por mes

Codigo:
```python
# Datos del 2019
talks_df_2019 = talks_df[talks_df['year'] == 2019]
# Variable para ordenar los meses
order = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December')
# Guardamos en una nueva columna la lista de meses en el orden correcto
talks_df_2019['order'] = pd.Categorical(talks_df_2019['month'], categories=order, ordered=True)
# Ordenamos por el orden de la lista y actualizamos el dataframe
talks_df_2019.sort_values(by=['order'], inplace=True)

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
```
''')
talks_df_2019 = talks_df[talks_df['year'] == 2019]
order = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December')
talks_df_2019['order'] = pd.Categorical(talks_df_2019['month'], categories=order, ordered=True)
talks_df_2019.sort_values(by=['order'], inplace=True)
mes2k19 = talks_df_2019.groupby('order')['title'].count()
fig, ax = plt.subplots(figsize=(30,15))
ax.bar(mes2k19.index, mes2k19.values)
ax.set_title('2019 - Charlas por Mes', fontsize=15)
ax.set_facecolor('white')
step_value=mes2k19.max()/20
for i, v in enumerate(mes2k19):
    ax.text(i, v+step_value/5, v, color='gray')
# Imprime el grafico en streamlit
st.pyplot(fig)
