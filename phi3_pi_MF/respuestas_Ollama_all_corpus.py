import requests
import json
import random
import time
import pickle
import pandas as pd

model1 = "dRITphi3_piN"

template = {
  "Answer": "",
  "Explanation":""
}

ruta="../../RIT_LLM/predictions/"

corpus=["RL_MultiNLI_DEVM_600","RL_MultiNLI_DEVMM_600","RL_SICK_DEV_600","RL_SICK_TEST_600",
        "RL_SNLI_DEV_600","RL_SNLI_TEST_600","RL_MultiNLI_TEST_600_2","RL_SICK_TEST_600_2","RL_SNLI_TEST_600_2"]

inicio = time.time()

for c in corpus:
    lista_respuestasOllama=[]
    df_t=pd.read_pickle(ruta+c+".pickle")
    for index,strings in df_t.iterrows():
        #print(strings["sentence1"],strings["sentence2"],strings['gold_label'])
        prompt = '''
        Analyze the following sentences: 
        Text: '''+strings["sentence_A"]+'''  
        Hypothesis: '''+ strings["sentence_B"]+'''
        with the next values for 15 features:
        sums:'''+ str(strings["sumas"])+'''
        negH:'''+ str(strings["negH"])+'''
        list_compability:'''+ str(strings["list_comp"])+'''
        list_incompability:'''+ str(strings["list_incomp"])+'''
        relation:'''+ str(strings["relation"])+'''
        entail:'''+ str(strings["entail"])+'''
        negT:'''+ str(strings["negT"])+'''
        mutinf:'''+ str(strings["mutinf_t"])+'''
        contradiction:'''+ str(strings["contradiction"])+'''
        no_matcheadas:'''+ str(strings["no_matcheadas"])+'''
        Jaro-Winkler_rit:'''+ str(strings["Jaro-Winkler_rit"])+'''
        simBoW:'''+ str(strings["simBoW"])+'''
        jaccard:'''+ str(strings["jaccard"])+'''
        overlap_ent:'''+ str(strings["overlap_ent"])+'''
        max_info_t:'''+ str(strings["max_info_t"])+'''    
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
            response = requests.post("http://localhost:11434/api/generate", json=data, stream=False,timeout=15)        
            json_data = json.loads(response.text)
            lista_respuestasOllama.append(json.dumps(json.loads(json_data["response"]), indent=2))
            print(index)
        except:
            print("Saltó",index,c)
            lista_respuestasOllama.append("NA")
        
    with open("resultados/rit_"+c+".pickle", "wb") as f:
        pickle.dump(lista_respuestasOllama, f)
    time.sleep(60)
    
fin = time.time()
print("Tiempo que se llevo:",round(fin-inicio,2)," segundos")
