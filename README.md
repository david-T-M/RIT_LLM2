# RIT_LLM2
Segundo experimento de RIT con Modelos de Lenguaje
En el promtp se coloca la información de cada par <T,H>
Así como la descripción de las 15 features

En este experimento solo le pedimos una etiqueta Entailment, Neutral o Contradiction con el modelo base
phi3/
    resultados/
    finales/
nohup ../../anaconda3/envs/rit/bin/python3 respuestas_Ollama_all_corpus.py > salida.txt &

En este experimento le proporcionamos información de nuestras features y para cada muestreo
le proporcionamos los valores de los features de <T,H>

phi3_pi/
    resultados/
    finales/
nohup ../../anaconda3/envs/rit/bin/python3 respuestas_Ollama_all_corpus.py > salida.txt &