import string
from nltk.corpus import stopwords


def Cleaner(text):
    text = text.lower()
    delete_dict = {sp_character: '' for sp_character in string.punctuation.replace('.', '')} 
    delete_dict[' '] = ' ' 
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    #print('cleaned:'+text1)
    text2 = text1.replace('newlinechar', '')
    text2 = text2.replace('\n', '')
    text2 = text2.replace('  ', '')
    return text2.lower()

stop_words = stopwords.words('english')
# function to remove stopwords
def remove_stopwords(text):
    textArr = text.split(' ')
    rem_text = " ".join([i for i in textArr if i not in stop_words])
    return rem_text


