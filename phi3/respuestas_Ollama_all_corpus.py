import requests
import json
import random
import time
import pickle
import pandas as pd

model1 = "phi3"

template = {
  "Answer": "",
  "Explanation":""
}

ruta="../../RIT_LLM/predictions/"

corpus=["RL_MultiNLI_DEVM_600","RL_MultiNLI_DEVMM_600","RL_Scitail_DEV_600","RL_Scitail_TEST_600",
        "RL_SICK_DEV_600","RL_SICK_TEST_600","RL_SNLI_DEV_600","RL_SNLI_TEST_600"]

for c in corpus:
    lista_respuestasOllama=[]
    df_t=pd.read_pickle(ruta+c+".pickle")
    for index,strings in df_t.iterrows():
        #print(strings["sentence1"],strings["sentence2"],strings['gold_label'])
        prompt = '''
        You are an expert in Recognition Textual Entailment over pairs of Text and a Hypothesis. 
        Answer if the Text entails Hypothesis, with only one of the following answers: Entailment or Neutral or Contradiction.
        Analyze the following sentences: 
        Text: '''+strings["sentence_A"]+'''  
        Hypothesis: '''+ strings["sentence_B"]+'''
        only Answer: Entailment or Neutral or Contradiction. Plus your Explanation.
        Use the following template:  '''+json.dumps(template)+''' do not modify the template'''
        #print(prompt)
        data = {
            "prompt": prompt,
            "model": model1,
            "format": "json",
            "stream": False,
            #"options": {"temperature": 0.2},
        }
        try:
            response = requests.post("http://localhost:11434/api/generate", json=data, stream=False,timeout=25)        
            json_data = json.loads(response.text)
            lista_respuestasOllama.append(json.dumps(json.loads(json_data["response"]), indent=2))
            print(index)
        except:
            print("Saltó",index,c)
            lista_respuestasOllama.append("NA")
        
    with open("resultados/rit_"+c+".pickle", "wb") as f:
        pickle.dump(lista_respuestasOllama, f)
    time.sleep(30)