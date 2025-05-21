# TF-IDF(Term Frequency -Inverse Document Frequency):
#CANALIZACION (pipeline) 
#Realizar una "canalizacion" o "pipeline" para realizar el siguiente CORPUS CORPUS.Lenguaje.text
#1-Aplicar stopwords
#2-lematizacion
#3-TF-IDF
#4-Mostrar el corpus PREPARADO
#5-Mostrar la MATRIZ  TF-IDF generada
#6-Mostar el vocabulario generado


#Librerias  necesarias

import nltk
from nltk.corpus  import stopwords, wordnet
from nltk.tokenize import word_tokenize   # ES UNA FUNCION QUE USA punkt, separa el texto en oraciones o palabras.
from nltk.stem import WordNetLemmatizer 

#Se descarga solo si es necesario (por unica vez) 

#nltk.download('punkt') Se usa para  separa texto en palabras o oraciones ,MODELO DE TOKENIZACION.
#nltk.download('stopwords')se elimina palabras simples del texto(the,is,and) que no sirven.
#nltk.download('wordnet') diccionario en ingles, Usa wordNetLematizer ,para convertir palabras a su forma base. LEMATIZACION.

# Corpus a analizar(texto original)

corpus  =[
"Python is an interpreted and high-level language, while CPlus is a compiled and low-level language .-",
"JavaScript runs in web browsers, while Python is used in various applications, including data science and artificial intelligence.",
"JavaScript is dynamically and weakly typed, while Rust is statically typed and ensures greater data security .-",
"Python and JavaScript are interpreted languages, while Java, CPlus, and Rust require compilation before execution.",
"JavaScript is widely used in web development, while Go is ideal for servers and cloud applications.",
"Python is slower than CPlus and Rust due to its interpreted nature."
"JavaScript has a strong ecosystem with Node.js for backend development, while Python is widely used in data science .-",
"JavaScript does not require compilation, while CPlus and Rust require code compilation before execution .-",
"Python and JavaScript have large communities and an extensive number of available libraries.",
"Python is ideal for beginners, while Rust and CPlus are more suitable for experienced programmers."
]

# Tokenizar (separa en palabras y pasamos a minusculas)
texto = ''.join(corpus)
tokens = word_tokenize(texto.lower())
print("_"*70)

# Elimino stopwords y signos de puntuacion.
stop_words = set(stopwords.words('english')) 
tokens_filtrados = [palabra for palabra in tokens if palabra in tokens if palabra.isalpha() and palabra not in stop_words]
print("_"*70)


#Inicia el lematizador
lemmatizer = WordNetLemmatizer()
tokens_lematizados = [lemmatizer.lemmatize(palabra) for palabra in tokens_filtrados]
print("_"*70)

print("original:",tokens)
print("_"*70)

print("tokens filtrados:",tokens_filtrados)
print("_"*70)

print("tokens lemantizados:",tokens_lematizados)
print("_"*70)



# Se prepara el corpus limpio
corpus_preparado =[] #Guarda el texto procesado

for texto in corpus:
    tokens = word_tokenize(texto.lower())  # Tokenizar y pasar a minúsculas
    tokens_filtrados = [word for word in tokens if word.isalpha() and word not in stop_words]  #  Filtrar
    tokens_lematizados = [lemmatizer.lemmatize(word) for word in tokens_filtrados]  #  Lematizar
    frase_limpia = " ".join(tokens_lematizados)  # Une en una oración
    corpus_preparado.append(frase_limpia)  # Guarda el resultado
    
    print("corpus_preparado:", corpus_preparado)
    print("_"*70)

# GENERAR LA MATRIZ TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer() #Transforma el texto en numeros segun la frecuencia e importancia de cada palabra.
tfidf_matrix = vectorizer.fit_transform(corpus_preparado) #Analiza el texto y calcula los valores.
print("_"*70)

# MOSTRAR LA MATRIZ
print("\nMatriz TF-IDF:\n")
print(tfidf_matrix.toarray()) #Muestra los resultados como matriz(filas=oraciones, columnas=palabras)
print("_"*70)


#Muestra el vocabulario generado
print("\nVocabulario generado:\n")
print(vectorizer.get_feature_names_out())#Muestra una lista de todas las palabras(vocabulario) que analizo.
print("_"*70)



# Analizar el mismo y redactar un informe con las conclusiones obtenidas
# 1-obtener las jeraquias de 6 palabras mas usadas en todos el Corpus.
# 2-la palabra menos ustilizada.
# 3-las palabras mas repetidas en la misma oracion.
# 4-imprimir el grafico de Distribucion de frecuencia.



from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Preparación
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Corpus original

corpus  =[
"Python is an interpreted and high-level language, while CPlus is a compiled and low-level language .-",
"JavaScript runs in web browsers, while Python is used in various applications, including data science and artificial intelligence.",
"JavaScript is dynamically and weakly typed, while Rust is statically typed and ensures greater data security .-",
"Python and JavaScript are interpreted languages, while Java, CPlus, and Rust require compilation before execution.",
"JavaScript is widely used in web development, while Go is ideal for servers and cloud applications.",
"Python is slower than CPlus and Rust due to its interpreted nature."
"JavaScript has a strong ecosystem with Node.js for backend development, while Python is widely used in data science .-",
"JavaScript does not require compilation, while CPlus and Rust require code compilation before execution .-",
"Python and JavaScript have large communities and an extensive number of available libraries.",
"Python is ideal for beginners, while Rust and CPlus are more suitable for experienced programmers."
]


# Lista para juntar todas las palabras lematizadas
tokens_lematizados = []

# Procesamiento de cada oración
for texto in corpus:
    tokens = word_tokenize(texto.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    lemas = [lemmatizer.lemmatize(t) for t in tokens]
    tokens_lematizados.extend(lemas)  # Agregamos a la lista general
print("_"*70)
    

# Contar palabras manualmente 
frecuencia = {}

for palabra in tokens_lematizados:
    if palabra in frecuencia:
        frecuencia[palabra] += 1
    else:
        frecuencia[palabra] = 1
print("_"*70)


# Ordenar por frecuencia descendente
frecuencia_ordenada = sorted(frecuencia.items(), key=lambda x: x[1], reverse=True)
print("_"*70)


# Mostrar las 6 palabras más frecuentes
print("Top 6 palabras más frecuentes:")
for palabra, cantidad in frecuencia_ordenada[:6]:
    print(f"{palabra}: {cantidad}")
print("_"*70)

    
# PALABRA MENOS USADAS
print("\npalabra menos utilizadas:")
print(frecuencia_ordenada[-1]) # indica la ultima palabra menos frecuente.




# Elegimos la oración que quiera usar.( EN ESTE CASO USE LA ORACION DOS)
oracion = corpus[1]

# Procesar la oración
tokens = word_tokenize(oracion.lower())
tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
lemas = [lemmatizer.lemmatize(t) for t in tokens]

# Contar repeticiones 
repeticiones = {}
for palabra in lemas:
    if palabra in repeticiones:
        repeticiones[palabra] += 1
    else:
        repeticiones[palabra] = 1

# Mostrar los resultados
print("\nPalabras repetidas en la oración seleccionada:")
for palabra, cantidad in repeticiones.items():
    print(f"{palabra}: {cantidad}")


#Realizare 3 modelos de graficos para realizar una comparacion en el informe.

#GRAFICO DE BARRAS

import matplotlib.pyplot as plt

palabras = [p[0] for p in frecuencia_ordenada[:10]]
valores = [p[1] for p in frecuencia_ordenada[:10]]

plt.figure(figsize=(10, 5))
plt.bar(palabras, valores, color='skyblue')
plt.title("Top 10 Palabras Más Frecuentes")
plt.xlabel("Palabras")
plt.ylabel("Frecuencia")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#GRAFICO DE TORTA

etiquetas = [p[0] for p in frecuencia_ordenada[:5]]
valores = [p[1] for p in frecuencia_ordenada[:5]]

plt.figure(figsize=(6,6))
plt.pie(valores, labels=etiquetas, autopct='%1.1f%%', startangle=90)
plt.title("Distribución de las 5 Palabras Más Frecuentes")
plt.show()

#GRAFICO DE LINEAS

palabras = [p[0] for p in frecuencia_ordenada[:10]]
valores = [p[1] for p in frecuencia_ordenada[:10]]

plt.figure(figsize=(10,5))
plt.plot(palabras, valores, marker='o', linestyle='-', color='green')
plt.title("Frecuencia de Palabras (Gráfico de Línea)")
plt.xlabel("Palabras")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()