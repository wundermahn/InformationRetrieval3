"""
 SECTION: Libraries
 These libraries are needed for program execution
 NLTK: Tokenization of the data
 String: String manipulation functions
 CSV: Export to CSV for testing
 Operator: For sorting through dicts
 Re: For regular expressions used to remove tags
 Collections: Used for dict creation and custom data structure creation
 Struct: Used for packaging data as a binary file
 zlib: Used for compressing data
 ast: Used for conversion of data
 io: For reading and writing binary data
 Sys: For passing arguments to the program
 time: For timing sections and portions of the code
 math: For the log base 2 function
 itertools: For the islice operation, used to retrieve entries from a dict
"""
import nltk, string, csv, operator, re, collections, sys, struct, zlib, ast, io, math, time
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from collections import defaultdict, Counter
from bs4 import BeautifulSoup as soup
from math import sqrt
from itertools import islice

"""
    SECTION: Functions
    These functions areused throughout the program and are housed in this section
    as a header of sorts. They make the actual logic and flow of the program much cleaner.
"""

# This function reads a file and returns a variable
def get_input(filepath):
    # Open the filepath specified
    f = open(filepath, 'r')
    # Read the file into memory and store it into a variable
    content = f.read()
    # Return the variable
    return content

# This function removes numbers from an array
def remove_nums(arr): 
    # Declare a regular expression
    pattern = '[0-9]'  
    # Remove the pattern, which is a number
    arr = [re.sub(pattern, '', i) for i in arr]    
    # Return the array with numbers removed
    return arr

# This function cleans the passed in paragraph and parses it
def get_words(para):   
    # Create a set of stop words
    stop_words = set(stopwords.words('english'))
    # Split it into lower case    
    lower = para.split()
    # Remove punctuations
    no_punctuation = (nopunc.translate(str.maketrans('', '', string.punctuation)) for nopunc in lower)
    # Remove integers
    no_integers = remove_nums(no_punctuation)
    # Remove stop words
    dirty_tokens = (data for data in no_integers if data not in stop_words)
    # Ensure it is not empty
    tokens = [data for data in dirty_tokens if data.strip()]
    # Ensure there is more than 1 character to make up the word
    tokens = [data for data in tokens if len(data) > 1]

    # This is the 5-stemming experiment
    tokens = [data[:5] for data in tokens]
    # This is the 5-stemming experiment
    
    # Return the tokens
    return tokens    

# This function converts a dict object to binary
# The struct page was utilized heavily: https://docs.python.org/2/library/struct.html
def pack_dict(mydict):
    # Turn the dict into a string
    s = str(mydict)
    # Turn the string dict into a list of bytes
    s = bytes(s, 'utf-8')
    # Compress that using zlib
    s = zlib.compress(s)
    # Take the length of that object, which is needed to pack it
    length = len(s)
    # Use struct.pack() to pack it into a binary file
    bs = struct.pack('%ds' %length, s)
        
    # Return the length, which is needed to unpack, and the binary file
    return length, bs 

# This function saves a file to binary
# NOTE - It should be noted that the disk location is hardcoded to emphasis that it IS being written to disk
# The movement to an argument would be trivial
def save_binary(length, binary):
# The following Stack Overflow post helped to understand how to read and write binary data.
#https://stackoverflow.com/questions/17349484/python-mangles-struct-pack-strings-written-to-disk    
    # Create an io object to write the binary file
    binary_out = io.open("C:\\Users\\Kelly\\Desktop\\Testing\\test.bin", "wb+")
    # Create an io object to write the length in binary
    length_out = io.open("C:\\Users\\Kelly\\Desktop\\Testing\\length.bin", "wb+")
    # Write the binary file
    binary_out.write(binary)

    # Pack the length integer into binary data
    length = struct.pack('i', length)
    # Write the binary length
    length_out.write(length)

    # No return type

# This function creates a lookup hash map of all IDF values for all terms
# in the dictionary
def create_idf_lookup(index, num_paragraphs):
    # Create a hash map to return
    idf_dict = {}
    # Pass in the inverted index
    for k, v in index.items():
        # The curr IDF value of the term
        currIDF = math.log2(num_paragraphs / v[0])    
        # Append the term and its IDF to the return dict
        values = idf_dict.setdefault(k,currIDF)

    # Return the dict
    return idf_dict       

# This returns the idf value for a specified term
def getIDF(idf, term):
    # Create a list of all matching items
    words = [v for k, v in idf.items() if k == term]
    # Just return the item, not in a list format
    idf = words[0]
    # Return it
    return idf

# This function calculates the length of a query or document
def calculateLength(myDict, idfDict):
    # Create a blank hash map to store the results
    new_dict = {}
    # Loop through the document or query
    for k, v in myDict.items():
        # Keep a running total of the number to be squared
        theSum = 0
        # Loop through each of the term frequencies of each word in the document or query
        for aTuple in v:
            # Take its TFxIDF value and square it, add it to the running sum
            theSum += (aTuple[1] ** 2)
        # Take the square root of the running sum
        theSum = sqrt(theSum)
        # Now you know the length of that document or query
        new_dict[k] = theSum
    
    # Return the dict
    return new_dict

# This function reads in a query
def read_query(filepath):
    # Open the specified filepath
    f = open(filepath, 'r')
    # Read the file into memory storing it in the content variable
    content = f.read()
    # Return the content variable
    return content

# This function processes the query and tokenizes it
def process_query(file):
    # Read in the query
    query = read_query(file)
    # Declare a regular expression for NLTK to tokenize the query
    p = r'<Q ID=\d+>(.*?)</Q>'
    # Declare the tokenizer    
    paras = RegexpTokenizer(p)   
    # Declare the current word count, which is how many times the word has appeared 
    # in the current document
    currWordCount = 0   
    # Temp list to hold the structure of [query_id, (word, doc_frequency)]
    currDocList = []
    # List to hold the list of doc lists to later be converted into a dictionary
    all_doc_lists = []
    # ID of the query
    query_id = 0

    # Begin to tokenize the query
    for para in paras.tokenize(query):             
        # Increase the ID (since they are iterative)
        query_id += 1
        # Convert to lowercase
        para = para.lower()
        # Get the words
        tokens = get_words(para)
        # Get the unique words and convert it back into a list
        individual_tokens = list(set(tokens))               
        # Now loop through each word
        for it in individual_tokens:        
            # Update the current word count for the current query  
            currWordCount = tokens.count(it)
            # If the query is not blank        
            if(currWordCount) > 0:
                # Create the current list showing the query ID, the word and its count
                currDocList = [query_id, tuple([it, currWordCount])]
                # Append that to the matter list
                all_doc_lists.append(currDocList)
            else:
                continue
    
    # Declare a new hash map for term freq by query
    termfreq_by_query = {}    

    # Loop through the list of lists
    for key, new_value in all_doc_lists:
        # Create the dict
        values = termfreq_by_query.setdefault(key, [])
        values.append(new_value)

    # Return the dict
    return termfreq_by_query  

# This function calculates the dot product of two passes in matrices
def dot_prod(source, target):
    # The function assumes there are dictionaries (hash maps) of equal length
    return sum(source.get(key, 0) * target.get(key, 0) for key in source.keys() | target.keys())

# This function sorts a dictionary object
def sort_dictionary(mydict):
    # This code was actually inspired by an article below:
    # Inspired by: https://www.w3resource.com/python-exercises/dictionary/python-data-type-dictionary-exercise-1.php
    # Order the dictionary in a list
    # It sorts the items in a (list of) tuple format in reverse (descending) order on the 1 index (frequency) item
    ordered_dict = sorted(mydict.items(), key=operator.itemgetter(1), reverse=True)
    # Turn it back into a dictionary
    dict(ordered_dict)

    # Return it
    return ordered_dict

# This function calculates the size of the vocabulary
def vocabulary_size(mydict):
    # Return the length of the dictionary passed in
    return len(mydict)

# This function takes the first n items of a dictionary
def take(n, iterable):
    #https://stackoverflow.com/questions/7971618/python-return-first-n-keyvalue-pairs-from-dict
    #Return first n items of the iterable as a dict
    return dict(islice(iterable, n))
    
### NEW
def driver(file):
    #t1 is the start of the file
    t1 = time.time()

    # Load the data from the text file
    myfile = get_input(file)
    # Create a regular expression to tokenize the paragraphs
    p = r'<P ID=\d+>.*?</P>'       
    # Create the tokenizer
    paras = RegexpTokenizer(p)   
    # Create a counter for the document frequency
    document_frequency = collections.Counter()   
    # Create a counter for the collection frequency
    collection_frequency = collections.Counter()   
    # Create a master list to hold the individual lists created for tuple of (docID, word)
    all_lists = []    
    # Instance of the current word count in the paragraph
    currWordCount = 0   
    # The current list for the tuple of (docID, word)
    currList = [] 
    # The current list for the term frequencies by document
    currDocList = []
    # The master list to hold the individual lists created for the TF by doc
    all_doc_lists = []
    # The total number of documents in the file
    num_paragraphs = len(paras.tokenize(myfile))  

    # Create a regular expression to find the ID of the document
    group_para_re = re.compile(r"<P ID=(\d+)>")
    # Tokenize the file, and begin going through each document
    for para in paras.tokenize(myfile):             
        # Grab the document ID
        group_para_id = group_para_re.match(para)        
        para_id = group_para_id.group(1)       
        # Convert the paragraph to lowercase
        para = para.lower()
        # Get all of the words / tokens in the paragraph
        tokens = get_words(para)
        # Update the total collections of tokens
        collection_frequency.update(tokens)             
        # Create a distinct or unique set of the tokens
        individual_tokens = list(set(tokens))
        # Update the document frequency for the terms
        document_frequency.update(set(tokens))         
        # This mimics the logic in the get_words function
        for it in individual_tokens:          
            # Grab the number of occurences of the specified word
            currWordCount = tokens.count(it)          
            # Update the current list of the form word: (doc_id, term_freq)
            currList = [it, tuple([para_id, currWordCount])]          
            # Append that current list to the master list of lists
            all_lists.append(currList)
            # Update the current list of the form doc_id: (word, term_freq)
            currDocList = [para_id, tuple([it, currWordCount])]
            # Append that current list to the master list of lists
            all_doc_lists.append(currDocList)

    # Create blank dictionaries (hash maps) for the postings list, and the TF-by-doc
    d = {}
    termfreq_by_doc = {}    
    # Build the postings list
    for key, new_value in all_lists:       
        values = d.setdefault(key, [])       
        values.append(new_value)
    # Build the tf-by-doc hash map
    for key, new_value in all_doc_lists:
        values = termfreq_by_doc.setdefault(key, [])
        values.append(new_value)

    # t2 is after the tokenization
    t2 = time.time()
    
    # Build the inverted index
    inverted_index = {word:(document_frequency[word], d[word]) for word in d}

    # t3 is after creating the index
    t3 = time.time()

    # Print out requested results
    print("Number of Paragraphs Processed: {}".format(num_paragraphs))
    print("Number of Unique Words (Vocabulary Size): {}".format(vocabulary_size(document_frequency)))
    print("Number of Total Words (Collection Size): {}".format(sum(collection_frequency.values())))
    print()

    # Build a hash map of word: tf-idf value
    idf_lookup = create_idf_lookup(inverted_index, num_paragraphs)
    #t4 is after creating the lookup
    t4 = time.time()

    # Create the TF-IDF by doc in a hashmap
    tfidf_by_doc = {a:[(c, int(idf_lookup[c]*d)) for c, d in b] for a, b in termfreq_by_doc.items()}
    #t5 is after creating the tfidf by doc
    t5 = time.time()

    # Calculate document lengths in a hashmap
    lengths = calculateLength(tfidf_by_doc, idf_lookup)    
    
     #Now, process the query:
     # Read in the query
    query_results = process_query("D:\\Grad School\\Fall 2019\\605.744.81.FA19 - Information Retrieval\\Programming Assignment 3\\cds14.txt\\all_query.txt")
    # query_results = process_query("D:\\Grad School\\Fall 2019\\605.744.81.FA19 - Information Retrieval\\Programming Assignment 3\\animal.topics.txt")
    # Create tf-idf for the query vector
    query_vector = {a:[(c, int(idf_lookup[c]*d)) for c, d in b] for a, b in query_results.items()}

    # Calculate the query lengths in a hash map
    queryLengthDict = calculateLength(query_vector, idf_lookup)
    
    # Loop through the queries
    for i in range(len(queryLengthDict)):
        # i is the query_id
        i += 1
        # Grab the current queries length
        currLength = [v for k, v in queryLengthDict.items() if k == i]
        # Put the length into a variable, not a list
        currLength = currLength[0]
        # Create a hash map of the results of the dot products between the query and all documents
        result = {}
        # Calculate the dot products
        for k, v in tfidf_by_doc.items():
            for se in query_vector.values():
                result[k] = dot_prod(dict(v), dict(se))
        
        # If its the first query, print it out
        if(i == 1):
            print("FIRST QUERY")
            print([v for k, v in query_vector.items() if k == i])
            print()

        # Calculate cosine similarities for each document and the query
        cosine_sims = {k:v / (lengths[k] * currLength) if lengths[k] != 0 else 0 for k, v in result.items()}   
        #t6 is grabbing the sorted sims
        t6 = time.time()
        # Sort the cosine similarities by their value
        sorted_sims = sort_dictionary(cosine_sims)
        # Grab the top 100 documents
        top_100 = take(100, sorted_sims)

        # Create a variable to hold the rank of each document
        rank = 0
        # Loop through the top 100 results
        for k, v in top_100.items():
            # Increasethe rank
            rank += 1
            # Print to file the required format
            print(str(i) + " " + "Q0" + " " + str(k) + " " +  str(rank) + " " + str(v) + " " + "tonykelly", file=open("C:\\Users\\Kelly\\Desktop\\kelly-a.txt", "a"))            
        # Print a space between each query
        print("\n", file=open("C:\\Users\\Kelly\\Desktop\\kelly-a.txt", "a"))
        
    # Calculate total run time for the program
    t7 = time.time()
    print()
    print()

    # Print out timing metrics
    print("TOTAL TIME TO COMPLETE: {}".format(t7-t1))
    print("TIME TO GRAB SORTED COSINE SIMS: {}".format(t6-t5))
    print("TIME TO CREATE TFIDF BY DOC: {}".format(t5-t4))
    print("TIME TO CREATE IDF LOOKUP: {}".format(t4-t3))
    print("TIME TO CREATE INVERTED INDEX: {}".format(t3-t2))
    print("TIME TO TOKENIZE: {}".format(t2-t1))

    # Alert end user to end of program
    print("PROGRAM COMPLETED")
  
# Main program
def main():
    
    # include hard code to file path
    driver("D:\\Grad School\\Fall 2019\\605.744.81.FA19 - Information Retrieval\\Programming Assignment 3\\cds14.txt\\big.txt")
    #driver("D:\\Grad School\\Fall 2019\\605.744.81.FA19 - Information Retrieval\\Programming Assignment 3\\animal.txt")

main()