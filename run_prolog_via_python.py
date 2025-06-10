from prologlike_new import *

#if the path of your Prolog file is for example C:\Users\YourProfile\Dekstop\test.pl,then write below r"C:\\Users\\YourProfile\\Dekstop\\test.pl"
#Av το μονοπάτι του αρχείου σας είναι π.χ.C:\\Users\\YourProfile\\Dekstop\\test.pl, τότε γράψτε r"C:\\Users\\YourProfile\\Dekstop\\test.pl"


selected_file = "type_your_filepath_as_indicated_above"
# Remove any non-printable characters from the file path, preventing unicode errrors 
# Αφαιρεση τυχόν μη εκτυπώσιμων χαρακτήρες από τη διαδρομή αρχείου,αποτρέποντας σφάλματα unicode
clean_path = ''.join(filter(str.isprintable, selected_file))
"""loading Prolog programs(rules and facts) from the external file/Φόρτωση προγραμμάτων Prolog(γεγονότα και κανόνες) από το εξωτερικό αρχείο"""
executor.load_facts_from_file(clean_path)
print("\n") #Change line for better readability/Αλλαγή γραμμής για καλύτερη αναγνωσιμότητα

"""loading Prolog queries from the external file/Φόρτωση ερωτημάτων Prolog από το εξωτερικό αρχείο"""
queries = load_queries_from_file(clean_path)




