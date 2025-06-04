from prologlike_new import *

#if the path of your Prolog file is for example C:\Users\YourProfile\Dekstop\test.pl,then write instead of "None" this: r"C:\Users\YourProfile\Dekstop\test.pl"
#Av το μονοπάτι του αρχείου σας είναι π.χ.C:\Users\YourProfile\Dekstop\test.pl, τότε γράψτε αντί για "None" αυτό :r"C:\Users\YourProfile\Dekstop\test.pl" 
selected_file = None
"""loading Prolog programs(rules and facts) from the external file/Φόρτωση προγραμμάτων Prolog(γεγονότα και κανόνες) από το εξωτερικό αρχείο"""
executor.load_facts_from_file(selected_file)
print("\n") #Change line for better readability/Αλλαγή γραμμής για καλύτερη αναγνωσιμότητα

"""loading Prolog queries from the external file/Φόρτωση ερωτημάτων Prolog από το εξωτερικό αρχείο"""
queries = load_queries_from_file(selected_file)



