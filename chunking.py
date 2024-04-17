import numpy as np
import spacy
import spacy_transformers;

upper_limit=1012
lower_limit=20




# Load the Spacy model
nlp = spacy.load('en_core_web_lg')
nlp.remove_pipe("lemmatizer")
nlp.add_pipe("lemmatizer", config={"mode": "lookup"}).initialize()

def process(text):
    doc = nlp(text)
    sents = list(doc.sents)
    vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])

    return sents, vecs

def cluster_text(sents, vecs, threshold):
    clusters = [[0]];
    print('\n\n')
    for i in range(1, len(sents)):
        dotted = np.dot(vecs[i], vecs[i-1])
        print('[', dotted, ']', end=" ")
        if dotted < threshold:
            clusters.append([])
        clusters[-1].append(i)
    print('\n\n')
    return clusters

def clean_text(text):
    # Add your text cleaning process here
    print('TEXT')
    print(text)
    print('\n\n')
    return text

def lower_chunk_text(cluster_txt, threshold):
    sents_div, vecs_div = process(cluster_txt)
    reclusters = cluster_text(sents_div, vecs_div, threshold)
            
    for subcluster in reclusters:
        div_txt = clean_text(' '.join([sents_div[i].text for i in subcluster]))
        div_len = len(div_txt)
        
        if div_len > upper_limit:
            return lower_chunk_text(cluster_txt, threshold=threshold+0.1)
    return sents_div, vecs_div, reclusters

def chunk_text(text):
    # Initialize the clusters lengths list and final texts list
    clusters_lens = []
    final_texts = []

    # Process the chunk
    threshold = 0.3
    sents, vecs = process(text)
    #  sents, vecs, reclusters = lower_chunk_text(text, 0.4)

    print(sents)

    # Cluster the sentences
    clusters = cluster_text(sents, vecs, threshold)

    print('CLUSTERS: ')
    print (clusters)
    print('\n')

    for cluster in clusters:
        cluster_txt = clean_text(' '.join([sents[i].text for i in cluster]))
        cluster_len = len(cluster_txt)

        # bufferString = ""
        
        # Check if the cluster is too short
        if cluster_len < lower_limit:
            # print('buffer string is 0: ', bufferString)
            # bufferString+=cluster_txt
            continue
            
        
        # Check if the cluster is too long
        elif cluster_len > upper_limit:
            print('cluster length too big')
            threshold = 0.5

           
            
            # lets not run it on individual text chunk
            # sents_div, vecs_div = process(cluster_txt)
            reclusters = cluster_text(sents_div, vecs_div, threshold)
            
            for subcluster in reclusters:
                div_txt = clean_text(' '.join([sents_div[i].text for i in subcluster]))
                div_len = len(div_txt)
                
                if div_len < lower_limit:
                    continue
                elif div_len > upper_limit:
                    continue
                

                clusters_lens.append(div_len)
                # print('buffer string is 1: ', bufferString)      
                final_texts.append(div_txt)
                  
        else:
            # print('buffer string is 2: ', bufferString)
            clusters_lens.append(cluster_len) # do not account buffer string into total length
            final_texts.append(cluster_txt)
    
    return final_texts