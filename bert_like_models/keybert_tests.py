from keybert import KeyBERT


# Bad for some reason
# from sentence_transformers import SentenceTransformer
# sentence_model = SentenceTransformer('ufal/robeczech-base')
# kw_model = KeyBERT(model=sentence_model)

# model="paraphrase-multilingual-MiniLM-L12-v2"
kw_model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")

doc = """ Poplatek za podání přihlášky užitného vzoru. 
Poplatek za přihlášku užitného vzoru. 
Kolik stojí podání přihlášky užitného vzoru?. 
Kolik stojí přihláška užitného vzoru?. 
Kolik stojí přihlášení užitného vzoru?. 
Sleva na správní poplatek při podání přihlášky užitného vzoru elektronicky. 
Sleva na správní poplatek při elektronickém podání přihlášky užitného vzoru. 
Možnosti zaplacení správního poplatku za podání přihlášky užitného vzoru. 
Poplatek za prodloužení platnosti zápisu užitného vzoru. 
Poplatek za prodloužení platnosti užitného vzoru. 
Jak mohu prodloužit platnost zápisu užitného vzoru?. 
jakým způsobem mohu prodloužit platnost zápisu užitného vzoru?. 
Jak  můžu prodloužit platnost zápisu užitného vzoru?. 
Jakým způsobem můžu prodloužit platnost zápisu užitného vzoru?. 
Jak  mám prodloužit platnost zápisu užitného vzoru?. 
Jakým způsobem mám prodloužit platnost zápisu užitného vzoru?. 
Doba platnosti zápisu užitného vzoru. 
Maximální doba platnosti zápisu užitného vzoru. 
Obnova platnosti užitného vzoru. 
Formulář pro obnovu platnosti užitného vzoru. 
Formulář prodloužení platnosti užitného vzoru. 
"""

keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), 
                                    #  highlight=True,
                                     use_mmr=True, 
                                     diversity=0.5,
                                     top_n=5,
                                     )
print(keywords)