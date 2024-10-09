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

inicio = time.time()

for c in corpus:
    lista_respuestasOllama=[]
    df_t=pd.read_pickle(ruta+c+".pickle")
    for index,strings in df_t.iterrows():
        #print(strings["sentence1"],strings["sentence2"],strings['gold_label'])
        prompt = '''
        You are an expert in Recognition Textual Entailment over pairs of Text and a Hypothesis. 
        Answer if the Text entails Hypothesis, with only one of the following answers: Entailment or Neutral or Contradiction.
        I have developed a classification of words relationships between T and H, (t,h). Where t and h are words en T and H.
        - Group 1 (G1) relationships: Relationships from tokens of T to to tokens of H that are of similarity or hyperonymy. These relationships are usually found in the pairs of T and H that are of T and H that are labeled as Entailment. For example: dog in T and animal in H.
        - Group 2 (G2) relations: relations from tokens of T to tokens of H that are contradiction or distinct or co-hyponym are contradiction. These relations usually occur in pairs of T and H that are labeled as Contradiction. For example, dog in T and cat in H.
        - Group 3 (G3) relations: relations from tokens of T to to tokens of H that are of specificity or hyponymy. These relationships usually occur in pairs of T and H that are labeled Neutral. For example, animal in T and dog in H.
        - Group 4 Relationships: Relationships from tokens of T to to tokens of H that are not identified.

        In addition to this I am going to provide you with additional information that you will occupy to make the decision.

        Description of Features:
        1. sums: Difference of ratio of total information in M to MR. Where M contains all cosine similarities of the words of H with respect to those of T. MR is a matrix reduced from M by removing the words of H that we found from G1, G2 and G3.
        2. netH: Negations found in H. If words are found to be negated and how many exists in Hypothesis.
        3. list_compatibility: Number of (t,h) in G1. Has values greater than zero.
        4. list_incompatibility: Number of (t,h) in G2. Has values greater than zero.
        5. relation: Potential class indicator (based on overlap_entities), where 2 indicates possible neutrality, 1 possible entailment and -1 contradiction.
        6. entail: Proportion of (t,h) (G1) of H in T. It has values between 0 and 1
        7. negT: Negations found in T. If words are found to be negated and how many exists in Text.
        8. mutinf: As well as the matrix M, we create a MatrixI that contains the proportion of Mutual Information information of tokens of T and H. To obtain each tuple value of (t,h), we do it through its WE of each word and we use the following formula for get Mutual Information of words embeddings
        9. contradiction: Proportion of entities (G3) of H that contradict each other in T.
        10. not_matcheadas: Proportion of entities in H not contained in T, where there is no relation between entities (G4).
        11. Jaro_rit: Similarity of T and H with distance and penalty. This measure is that of Jaro Similarity for text strings, only adjusted for sentences where the match is over G1. It has values between 0 and 1, where one means that they are very similar and 0 not similar.
        12. simBoW: Cosine similarity of representation vector in BoW of T and H.
        13. jaccard: Proportion of words of H contained in T. It has values between 0 and 1.
        14. overlap_entities: Proportion of entities of H contained in T, where it is satisfied that the attributes of the entities of H are also found in T. It has values between 0 and 1, where 0 means that entities with their respective attributes of the hypothesis are not contained in entities of the Text.
        15. max_info_total: Proportion of information of tokens of T and H in M. For each token of H with respect to all those of T, the cosine similarity is obtained, with respect to its Word Embeddings of the word lemma, then the maximum values of each token of H with respect to those of T are summed.
        
        Now analyze the following sentences: 
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
    
    
fin = time.time()
print("Tiempo que se llevo:",round(fin-inicio,2)," segundos")
