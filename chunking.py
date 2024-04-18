import numpy as np
import spacy

upper_limit=1024
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
    clusters = [[0]]
    print("\n\n")
    for i in range(1, len(sents)):
        dotted = np.dot(vecs[i], vecs[i-1])
        print('[', dotted, ']: ', sents[i], end=" ")
        if dotted < threshold:
            clusters.append([])
        clusters[-1].append(i)
    print("\n\n")
    return clusters

def clean_text(text):
    # Add your text cleaning process here
    return text


# run this one time during initial run
def run_manual_chunk(cluster_txt):
    sents_div, vecs_div = process(cluster_txt)
    print("sentences: ", sents_div)
    print("\n\n")
    short_sentence = []
    all_sentences = []

    for i in range(len(sents_div)):
        sent = sents_div[i].text.strip()
        all_sentences.append(sent)
        sent = sent.split()
        if len(sent) < 9:
            print('short sentence: ', sent)
            short_sentence.append(i)

    
    # combine the sections and manually cluster
    sub_divided_sents = []
    sub_divided_vecs = []
    total_short_sentences = len(short_sentence)
    print("total short sentences: ", total_short_sentences, ' with indicies: ', short_sentence)
    for i in range(0, total_short_sentences):
        cur_index = short_sentence[i]
        next_index = cur_index
        if i < total_short_sentences -1:
            next_index = short_sentence[i+1]
        else: next_index = None
        if (next_index):
            sub_divided_sents += [all_sentences[cur_index: next_index]]
            sub_divided_vecs += [vecs_div[cur_index: next_index]]
        else:
            sub_divided_sents += [all_sentences[cur_index:]]
            sub_divided_vecs += [vecs_div[cur_index:]]

    # merge all indicies with a single element with the next
    to_delete = []

    for i in range(1, len(sub_divided_sents)):
        prev_division = sub_divided_sents[i-1]
        curr_division = sub_divided_sents[i]
        prev_division_len = len(prev_division)
        print("previous division len: ", prev_division_len)
        if (prev_division_len < 2):
            sub_divided_sents[i] = prev_division + curr_division
            sub_divided_vecs[i]  = sub_divided_vecs[i-1] + sub_divided_vecs[i]
            to_delete.append(i-1)

    # delete all merged indicies
    for i in to_delete: 
        del sub_divided_sents[i]
        del sub_divided_vecs[i]

    #combine each array into tasks
    combined = []
    for i in range(0, len(sub_divided_sents)):
        # joined = ''.join(sub_divided_sents[i])
        combined.append(''.join(sub_divided_sents[i]))

    for combination in combined:
        print('combined: ', combination, "\n")

    return combined


# break it down into as many pieces
def lower_chunk_text(text, threshold):
    sents_div, vecs_div = process(text)
    reclusters = cluster_text(sents_div, vecs_div, threshold)
    final_texts = []
            
    for subcluster in reclusters:
        div_txt = clean_text(' '.join([sents_div[i].text for i in subcluster]))
        div_len = len(div_txt)
        
        if div_len > upper_limit:
            return lower_chunk_text(text, threshold=threshold+0.1)
        else: final_texts.append(div_txt)

    return final_texts



def empty_buffer(final_chunks, buffer):
    if (len(buffer)> 0):
        final_chunks.append(buffer)
        buffer=""
    return final_chunks, buffer

def empty_buffer_if_needed(final_chunks, buffer, last_chunk):
    if (len(buffer)> 0):
        final_chunks.append(buffer)
        buffer=last_chunk
    return final_chunks, buffer


def empty_buffer_if_needed_1(final_chunks, buffer, final_texts, last_chunk):
    if (len(final_texts) > 1):
        # print("EMPTYING BUFFER: ", buffer, "\n\n")

        final_chunks.append(buffer)
        buffer=last_chunk
    else:
        buffer = final_texts[0]
    return final_chunks, buffer

# the initial method is to lower the overall length of a chunk. Some chunks may become too small. So reshuffle.
def recluster_small_chunks(chunks):
    final_chunks=[]
    buffer = chunks[0] # a combined string of chunks
    for i in range(1, len(chunks)):
        combined = buffer+chunks[i]
        if (len(combined) < upper_limit):
            final_texts = lower_chunk_text(combined, 0.4)
            final_chunks, buffer = empty_buffer_if_needed_1(final_chunks, buffer, final_texts, chunks[i])
        else:
            final_chunks, buffer = empty_buffer(final_chunks, buffer)
            buffer=chunks[i]
            
    final_chunks, buffer = empty_buffer(final_chunks, buffer)

    return final_chunks

def chunk_text(text):
    # Initialize the clusters lengths list and final texts list

    final_texts = []

    # Process the chunk
    threshold = 0.7

    combined  = run_manual_chunk(text)
    texts = []
    # for i in range(0, len(subdivided_sents_div)):
    final_texts = lower_chunk_text(text, threshold)
    texts += final_texts
    # sents, vecs = process(text)
    

    # print('CLUSTERS:\n\n')
    # print(reclusters)
    print('\n\n')

    # new_chunks = recluster_small_chunks(final_texts)

    return combined