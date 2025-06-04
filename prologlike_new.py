from typing import List, Union
import re


def split_args(arg_str: str):
        """Σπάει το '25, date(1,2,3), foo(bar)' σε ['25', 'date(1,2,3)', 'foo(bar)']"""
        parts = []
        depth = 0
        buff = ""
        for ch in arg_str:
            if ch == ',' and depth == 0:
                parts.append(buff.strip())
                buff = ""
            else:
                buff += ch
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
        if buff.strip():
            parts.append(buff.strip())
        return parts






def fresh_copy(goal, env=None):
    if env is None:
        env = {}
    if isinstance(goal, Variable):
        if goal in env:
            return env[goal]
        else:
            new_var = Variable(goal.name)
            env[goal] = new_var
            return new_var
    elif isinstance(goal, Constant):
        return Constant(goal.value)
    elif isinstance(goal, Predicate):
        new_terms = [fresh_copy(t, env) for t in goal.terms]
        return Predicate(goal.name, new_terms)
    elif isinstance(goal, Term):
        new_args = [fresh_copy(t, env) for t in goal.args]
        return Term(goal.functor, new_args)
    elif isinstance(goal, NotInstruction):
        return NotInstruction(fresh_copy(goal.goal, env))
    elif isinstance(goal, AndInstruction):
        return AndInstruction([fresh_copy(g, env) for g in goal.goals])
    elif isinstance(goal, OrInstruction):
        return OrInstruction([fresh_copy(g, env) for g in goal.goals])
    else:
        return goal


def tokenize(s: str):
    """
    Tokenizes the query string.
    Recognizes:
      - the negation operator: \+
      - punctuation: commas, semicolons, parentheses
      - identifiers (including variables and constants)
    """
    pattern = (
    r"(:\-)|"           # ‘:-’ for rule heads
    r"'[^']*'|"         # quoted atoms
    r"\\\+|"            # negation operator ‘\+’
    r"=:=|"             # numeric‐equality operator
    r"==|@<|=|is|"      # ‘==’, term‐ordering ‘@<’, plain ‘=’, and ‘is’
    r"(?:=<|>=|<|>)|"   # numeric comparisons ‘=<’, ‘>=’, ‘<’, ‘>’
    r"[,;()!]|"         # punctuation
    r"[A-Za-z0-9_+\-\*\/]+"   # identifiers, numbers’
)


    tokens = []
    for match in re.finditer(pattern, s):
        token = match.group(0)
        tokens.append(token)
    return tokens

def parse_query(query_str: str):
    tokens = tokenize(query_str)
    pos = 0
    var_dict = {}  # Shared dictionary for the entire query

    def parse_expr():
        return parse_disjunction()

    def parse_disjunction():
        nonlocal pos
        left = parse_conjunction()
        while pos < len(tokens) and tokens[pos] == ';':
            pos += 1
            right = parse_conjunction()
            if isinstance(left, OrInstruction):
                left.goals.append(right)
            else:
                left = OrInstruction([left, right])
        return left

    def parse_conjunction():
        nonlocal pos
        left = parse_infix_expr()
        while pos < len(tokens) and tokens[pos] == ',':
            pos += 1
            right = parse_infix_expr()
            if isinstance(left, AndInstruction):
                left.goals.append(right)
            else:
                left = AndInstruction([left, right])
        return left

    def parse_primary():
        nonlocal pos
        if tokens[pos] == r'\+':
            pos += 1
            # Καλούμε parse_infix_expr ώστε να διαβάσουμε ολόκληρη την έκφραση (π.χ. Y1==Y2)
            sub = parse_infix_expr()
            return NotInstruction(sub)
        elif tokens[pos] == '(':
            pos += 1
            expr = parse_expr()
            if pos >= len(tokens) or tokens[pos] != ')':
                raise Exception("Missing closing parenthesis")
            pos += 1
            return expr
        else:
            return parse_predicate()

        
    def parse_infix_expr():
        nonlocal pos
        left = parse_primary()
        # While the next token is an infix operator ("==" or "@<")
        while pos < len(tokens) and tokens[pos] in ("=","=:=","is", "==", "@<", "<", ">", "=<", ">="):
            op = tokens[pos]
            pos += 1
            right = parse_primary()
            # Build a predicate with operator op and two arguments.
            left = Predicate(op, [left, right])
        return left

    


    def parse_predicate():
        nonlocal pos, var_dict
        token = tokens[pos]
        if token.startswith('_'):
            pos += 1
            if token not in var_dict:
                var_dict[token] = Variable(token)
            return var_dict[token]
        if token.isdigit():
            pos += 1
            return Constant(token)
        pos += 1
        if (token == "true" or token == "fail") and (pos >= len(tokens) or tokens[pos] != '('):
            return Predicate(token, [])
        if token == "nl" and (pos >= len(tokens) or tokens[pos] != '('):
            return Predicate("nl", [])
        if token.startswith("'") and token.endswith("'"):
            return Constant(token[1:-1])
        if pos >= len(tokens) or tokens[pos] != '(':
            if token == '_':
                return Variable('_')
            if token[0].isupper():
                if token in var_dict:
                    return var_dict[token]
                else:
                    var = Variable(token)
                    var_dict[token] = var
                    return var
            else:
                return Constant(token)

        # Διαφορετικά, είναι κατηγόρημα με ορίσματα
        pred_name = token
        args = []
        pos += 1  # skip '('
        # Εδώ κάνουμε αναδρομικό parse για κάθε όρισμα
        while pos < len(tokens) and tokens[pos] != ')':
            arg = parse_primary()
            args.append(arg)
            # αν μετά το υπο‐όρισμα υπάρχει κόμμα, το προσπερνάμε
            if pos < len(tokens) and tokens[pos] == ',':
                pos += 1
        if pos >= len(tokens) or tokens[pos] != ')':
            raise Exception("Missing closing parenthesis for predicate")
        pos += 1  # skip ')'
        return Predicate(pred_name, args)


    return parse_expr()





# Κλάση για Μεταβλητές
class Variable:

    def __init__(self, name: str):
        self.name = name
        self.value = None  # Αρχική τιμή None για ανενοποίητη μεταβλητή
    
    def __repr__(self):
        return f"Variable({self.name})" if self.value is None else f"Variable({self.name}={self.value})"
    
    def unify(self, other):
        # Εάν η μεταβλητή έχει ήδη μια τιμή, ενοποίησε την τιμή της
        if self.value is not None:
            return self.value.unify(other)
        # Αν είναι μεταβλητή και δεν έχει ενοποιηθεί, δέχεται την τιμή του άλλου
        elif isinstance(other, Variable) and other.value is None:
            self.value = other
            return True
        else:
            self.value = other
            return True

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.name == other.name
        return False
    

class Constant:
    def __init__(self, value: Union[int, str]):
        self.value = value

    def __repr__(self):
        return f"Constant({self.value})"

    def unify(self, other):
        # Οι σταθερές ενοποιούνται μόνο αν είναι ίδιες
        #Constants are being unified onlyif they are same
        if isinstance(other, Constant):
            return self.value == other.value
        elif isinstance(other, Variable):
            return other.unify(self)
        return False
    

#Class for Terms  
# Κλάση για Όρους
class Term:
    def __init__(self, functor: str, args: List[Union[Variable, Constant, 'Term']]):
        self.functor = functor
        self.args = args

    def __repr__(self):
        args_str = ", ".join(map(str, self.args))
        return f"Term({self.functor}, [{args_str}])"
    
    def unify(self, other):
        """Controls if the predicate can be unified properly."""
        """Ελέγχει αν το κατηγόρημα μπορεί να ενοποιηθεί σωστά."""
        if not isinstance(other, Predicate) or self.name != other.name:
            return False
        if len(self.terms) != len(other.terms):
            return False
        return all(term1.unify(term2) for term1, term2 in zip(self.terms, other.terms))
#Class for Predicates
# Κλάση για Κατηγορήματα
class Predicate:
    def __init__(self, name: str, terms: List[Union[Variable, Constant, Term]]):
        self.name = name
        self.terms = terms

    def __repr__(self):
        terms_str = ", ".join(map(str, self.terms))
        return f"Predicate({self.name}, [{terms_str}])"
    
    def unify(self, other):
        if not isinstance(other, Predicate) or self.name != other.name or len(self.terms) != len(other.terms):
            return False
        return all(term1.unify(term2) for term1, term2 in zip(self.terms, other.terms))

            
    

        

class NotInstruction:
    """Represents the logic "NOT" in WAM"""
    """ Εκπροσωπεί την άρνηση ενός στόχου στη WAM """
    def __init__(self, goal):
        self.goal = goal

    def __repr__(self):
        return f"Not({self.goal})"


class AndInstruction:
    """Represents the logic "AND"""
    """ Αναπαριστά το λογικό και """
    def __init__(self, goals):
        self.goals = goals

    def __repr__(self):
        return f"And({', '.join(map(str, self.goals))})"

class OrInstruction:
    """ Αναπαριστά το λογικό ή """
    def __init__(self, goals):
        self.goals = goals

    def __repr__(self):
        return f"Or({', '.join(map(str, self.goals))})"

#Basic WAM instructions for represenation
# Βασικές εντολές WAM για την αναπαράσταση οδηγιών

class WAMInstruction:
    def __init__(self, opcode, *operands):
        self.opcode = opcode
        self.operands = operands

    def __repr__(self):
        return f"{self.opcode} " + " ".join(map(str, self.operands))

#######################ΜCompiler/Mεταγλωτιστής##################################################################################################################################

class WAMCompiler:
    def __init__(self):
        self.instructions = []  # Save of the instrcutions that are being produced from the wam/Αποθήκευση των παραγόμενων εντολών
    
    def reset(self):           #We garantee that WAMCompiler can be start from an empty state when new data must be transformed into WAM instructions
        self.instructions = [] #εξασφαλίζουμε ότι ο μεταγλωττιστής (WAMCompiler) μπορεί να ξεκινήσει από μια καθαρή κατάσταση όταν χρειάζεται να μεταγλωττίσει νέα δεδομένα ή ερωτήματα χωρίς να επηρεάζονται από προηγούμενες μεταγλωττίσεις.
    
    def compile_fact(self, predicate):
        self.instructions.append(WAMInstruction("get_structure", predicate.name, "A0"))
        """
        Compiing a Prolog fact in WAM instructions
        Μεταγλωττίζει ένα γεγονός Prolog σε εντολές WAM.
        """
        for i, term in enumerate(predicate.terms):
            if isinstance(term, Constant):
                # `put_constant`
                self.instructions.append(WAMInstruction("put_constant", term.value, f"X{i}"))
            elif isinstance(term, Variable):
                # `put_variable`
                self.instructions.append(WAMInstruction("put_variable", term.name, f"X{i}"))

        #End of execution with 'proceed' instruction
        # Τερματισμός με την εντολή `proceed`
        self.instructions.append(WAMInstruction("proceed"))
    
    def compile_rule(self, head, body):
        """
        Compiling a Prolog rule in WAM instructions
        Μεταγλωττίζει έναν κανόνα Prolog σε εντολές WAM.
        
        """
        self.instructions.append(WAMInstruction("allocate", len(head.terms)))
        self.instructions.append(WAMInstruction("get_structure", head.name, "A0")) 
        

        #First , we manipulate the head of the rule
        # Πρώτα, χειριζόμαστε το head του κανόνα
        for i, term in enumerate(head.terms):
            if isinstance(term, Constant):
                self.instructions.append(WAMInstruction("get_constant", term.value, f"X{i}"))
            elif isinstance(term, Variable):
                self.instructions.append(WAMInstruction("get_variable", term.name, f"X{i}"))
        #Then, we compile the predicates of the body part of the rule
        # Στη συνέχεια, μεταγλωττίζουμε τα κατηγορήματα του body του κανόνα
        for predicate in body:
            self.instructions.append(WAMInstruction("call",predicate.name,len(predicate.terms)))
            for i, term in enumerate(predicate.terms):
                if isinstance(term, Constant):
                    self.instructions.append(WAMInstruction("put_constant", term.value, f"X{i}"))
                elif isinstance(term, Variable):
                    self.instructions.append(WAMInstruction("put_variable", term.name, f"X{i}"))

        self.instructions.append(WAMInstruction("deallocate"))
        self.instructions.append(WAMInstruction("proceed"))
 

    def get_instructions(self):
        return self.instructions

###################################################Executor/Εκτελεστής##################################################################################################################
class WAMExecutor:
    def __init__(self):
        self.reset_state()
        self.rules = []
        self.cut_flag = False 
    def  reset_state(self):
        self.registers = {}  # Save the values of the registers/Αποθήκευση των τιμών στους καταχωρητές
        self.choice_points = []  # Array for saving choice points/Πίνακας για τα σημεία επιλογής
        self.state_stack = []  # State stack for backtracking/Στοίβα καταστάσεων για backtracking
        self.call_stack = []  # Call stack for the instructions/Στοίβα για παρακολούθηση των εντολών κλήσης
        self.heap = []  # Aray for heap/Πίνακας για τις αποθηκευμένες δομές (όρους)
        self.call_frames = []
        self.frame_pointer = None
        self.index = 0  # Initializing the pointer that indicates a Wam Instruction/Αρχικοποίηση του δείκτη index, ο οποίος δείχνει σε μια WAM Εντολή
        self.knowledge_base = [] #Πίνακας για αποθήκευση γεγονότων/Array for 
        self.rules = []   
        self.cut_flag = False       # Flag για λειτουργία cut αναδρομής(αν γίνει true  ενεργποιείται η λειτουργία cut και η οπισθοδρόμηση σταματά )
                                    #Flag for cut mode(if the flag is true then the cut mode is activated and backtracking stops)

    def load_facts(self, facts):

        """
        The facts are loaded in the array knowledge base 
        Φορτώνει τα γεγονότα σto σχετικό πίνακα."""
        self.knowledge_base = facts
        
    

    def load_facts_from_file(self, filename):
        '''Facts and rules are loaded from a .pl file, '''
        """Φορτώνει facts και rules από ένα αρχείο .pl,."""
        import re

        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        #Deleting Prolog comments (%... until the end of a line)
        # Αφαιρούμε σχόλια (% ... μέχρι το τέλος γραμμής)
        text = re.sub(r'%.*', '', text)
        #Splitting the text when a "dot" is being detected
        # Σπάμε σε δηλώσεις με βάση την τελεία
        stmts = re.split(r'\.\s*', text)

        print("Here is the content that  is loaded from the external file:")
        print(text.strip())


        facts = []
        rules = []
        for stmt in stmts:
            stmt = stmt.strip()
            if not stmt:
                continue
            # Rule/Κανόνας
            if ':-' in stmt:
                head_str, body_str = map(str.strip, stmt.split(':-', 1))
                head = parse_query(head_str)
                body_ast = parse_query(body_str)
                body = body_ast.goals if isinstance(body_ast, AndInstruction) else [body_ast]
                rules.append((head, body))
            else:
                # Fact/Γεγονός
                m = re.match(r'^([a-z]\w*)\s*\(\s*(.*)\s*\)$', stmt)
                if not m:
                    continue
                pred_name, arglist = m.group(1), m.group(2)
                raw_args = split_args(arglist)
                terms = []
                for a in raw_args:
                    if '(' in a:
                        nested = parse_query(a)
                        if not isinstance(nested, Predicate):
                            raise ValueError(f"Το '{a}' δεν έγινε Predicate")
                        terms.append(nested)
                    elif a == '_':
                        terms.append(Variable('_'))
                    elif a and a[0].isupper():
                        terms.append(Variable(a))
                    else:
                        terms.append(Constant(a))
                facts.append(Predicate(pred_name, terms))

        #Facts and rules are being detected in executor
        # Φορτώνουμε facts και rules στον executor
        self.load_facts(facts)
        self.rules = rules



    
    def match_fact(self, query):
        """Controlling if query is in knowledge_base array"""
        """Ελέγχει αν το query υπάρχει στo πίνακα των γεγονότων."""
        return any(fact.unify(query) for fact in self.knowledge_base)
        
        """Εxecuting a fact either is a single predicate or is a logical clause
         Εκτέλεση ενός γεγονότος είτε αυτό είνια ένα απλό κατηγόρημα είτε είναι μια λογική πρόταση"""

    def execute_goal(self, goal):
        if isinstance(goal, Predicate):
            return self.execute_query(goal)
        elif isinstance(goal, NotInstruction):
            return self.execute_not(goal)
        elif isinstance(goal, AndInstruction):
            return self.execute_and(goal)
        elif isinstance(goal, OrInstruction):
            return self.execute_or(goal)
        else:
            raise ValueError("Unknown goal type")

    def execute_query(self, query_predicate):

        

        # οnly succeeds if there is at least one fact or rule/επιτυχία αν υπάρχει μόνον τουλάχιστον ενας κανόνας ή γεγονός
# with the same name AND the same number of arguments:/ με το ίδιο όνομα ΚΑΙ τον ίδιο αριθμό ορισμάτων 
        if isinstance(query_predicate, Predicate) and all(isinstance(t, Variable) and t.name == '_' for t in query_predicate.terms):
            has_fact = any(
            f.name == query_predicate.name
            and len(f.terms) == len(query_predicate.terms)
            for f in self.knowledge_base
        )
        # and also in the rule heads/ομοίως  και για τις κεφαλές των κανόβν 
            has_rule = any(
                head.name == query_predicate.name
                and len(head.terms) == len(query_predicate.terms)
                for head, _ in self.rules
            )
            return True if (has_fact or has_rule) else False
    



         #Handling of built-in instructions neither they are call as a query or inside a rule 
          # Διαχείριση των built‐in εντολών όταν καλούνται είτε ως query είτε μέσα σε κανόνα.
        if query_predicate.name == "write":
            output = ""
            for term in query_predicate.terms:
                if isinstance(term, Constant):
                    output += str(term.value)
                elif isinstance(term, Variable):
                    if term.value is not None and isinstance(term.value, Constant):
                        output += str(term.value.value)
                    else:
                        output += term.name
                else:
                    output += str(term)
            #Printing the value(with out newline, because there is the nl-instruction which is responsible for this,look in  the "(*)" below)        
            # Εκτύπωση της τιμής (χωρίς αλλαγή γραμμής, η οποία αν χρειάζεται θα καλείται ξεχωριστά από nl, κοιτάμε το "(*)" πιο κάτω)
            print(output, end="")
            return True  # Returning true for success indication/επιστρέφουμε True για ένδειξη επιτυχίας

        elif query_predicate.name == "tab":
            num_spaces = 0
            if query_predicate.terms:
                term = query_predicate.terms[0]
                if isinstance(term, Constant):
                    try:
                        num_spaces = int(term.value)
                    except:
                        num_spaces = 0
                elif isinstance(term, Variable):
                    if term.value is not None and isinstance(term.value, Constant):
                        num_spaces = int(term.value.value)
                    else:
                        num_spaces = 0
            print(" " * num_spaces, end="")
            return True

        elif query_predicate.name == "nl":
            print()  # (*)
            return True

        elif query_predicate.name == "true":
            return True

        elif query_predicate.name == "fail":
            return False

        if query_predicate.name == "!":
            self.cut_flag = True
            self.choice_points.clear()  # Clearing the choice points/Καθαρίζουμε τα σημεία επιλογής
            return True
        

        
        # Handle built-in equality: "=="/Xειρισμός του τελεστή απόλυτης ισότητος "=="
        if query_predicate.name == "==":
            term1, term2 = query_predicate.terms
            # If both are constants, compare their values/Αν είναι σταθερές συγκρίνουμε τις τιμές τους.
            if isinstance(term1, Constant) and isinstance(term2, Constant):
                return term1.value == term2.value
            # If both are variables/Αν είναι μεταβλητές:
            if isinstance(term1, Variable) and isinstance(term2, Variable):
                # If both variables have been bound, compare the bound constants.
                if term1.value is not None and term2.value is not None:
                    if isinstance(term1.value, Constant) and isinstance(term2.value, Constant):
                        return term1.value.value == term2.value.value
                    else:
                        return term1.value == term2.value
                else:
                    # If either variable is still unbound, then we cannot claim equality.
                    return False
            # If one is a variable that is already bound and the other is a constant, compare.
            if isinstance(term1, Variable) and term1.value is not None and isinstance(term1.value, Constant) and isinstance(term2, Constant):
                return term1.value.value == term2.value
            if isinstance(term2, Variable) and term2.value is not None and isinstance(term2.value, Constant) and isinstance(term1, Constant):
                return term2.value.value == term1.value
            return False

        # Handle numeric comparisons
        if query_predicate.name in ("<", ">", "=<", ">="):
            t1, t2 = query_predicate.terms
            def get_num(t):
                if isinstance(t, Constant):
                    return int(t.value)
                if isinstance(t, Variable) and isinstance(t.value, Constant):
                    return int(t.value.value)
                return None

            v1 = get_num(t1)
            v2 = get_num(t2)
            if v1 is None or v2 is None:
                return False
            if query_predicate.name == "<":
                return v1 < v2
            if query_predicate.name == ">":
                return v1 > v2
            if query_predicate.name == "=<":
                return v1 <= v2
            if query_predicate.name == ">=":
                return v1 >= v2



        # Handle built-in term ordering: "@<"
        if query_predicate.name == "@<":
            term1, term2 = query_predicate.terms
            if isinstance(term1, Constant) and isinstance(term2, Constant):
                return term1.value < term2.value
            return False



        # συγκέντρωσε ΟΛΕΣ τις μεταβλητές (συμπεριλαμβανομένων των '_')
        all_vars = self._collect_variables(query_predicate)
        # κράτα μόνο αυτές που δεν είναι anonymous ('_')
        named_vars = [v for v in all_vars if v.name != '_']

        fact_solutions = self.find_all_solutions(query_predicate)
        rule_solutions = self.find_all_rule_solutions(query_predicate)
        all_solutions = fact_solutions + rule_solutions
        # αν έχουμε ΚΑΠΟΙΑ named variables, επιστρέφουμε τη λίστα με τις λύσεις
        if named_vars:
            return all_solutions if all_solutions else False

        # αλλιώς (μόνο anonymous vars ή καθόλου vars), απλό true/false
        return True if (all_solutions or self.match_fact(query_predicate)) else False




   

    def execute_and(self, and_instruction):
        
        # Διατηρούμε απλώς τα πηγαία goals, με το sharing του περιβάλλοντος
        goals = and_instruction.goals

        def solve_goals(goals, bindings):
            # Αν έχει συμβεί cut, σταματάμε αμέσως
            if self.cut_flag:
                return [bindings]
            # Όταν δεν έχει μείνει άλλο goal, έχουμε μια λύση
            if not goals:
                return [bindings]

            # Παίρνουμε τον πρώτο goal και τα υπόλοιπα
            goal = goals[0]
            remaining = goals[1:]

            # **ΔΕΝ** κάνουμε fresh_copy εδώ!
            # φρέσκο αντίγραφο κάναμε ήδη (ή δεν χρειάζεται) στο find_all_rule_solutions
            bound_goal = self._apply_bindings(goal, bindings)

            sols = []
            # Χειρισμός του cut (!) αν χρειάζεται
            if isinstance(goal, Predicate) and goal.name == "!":
                self.cut_flag = True
                self.choice_points.clear()
                return solve_goals(remaining, bindings)

            # Αν είναι αρνητικό goal
            if isinstance(goal, NotInstruction):
                result = self.execute_not(bound_goal)
                if result:
                    sols = [bindings]
                else:
                    sols = []
            else:
                # κανονικό predicate ή σύνθετο And/Or
                result = (self.execute_query(bound_goal)
                          if isinstance(bound_goal, Predicate)
                          else self.execute_and(bound_goal)
                          if isinstance(bound_goal, AndInstruction)
                          else self.execute_or(bound_goal))
                if isinstance(result, list):
                    for r in result:
                        new_bind = bindings.copy()
                        ok = True
                        for var, val in r.items():
                            if var in new_bind and new_bind[var] != val:
                                ok = False
                                break
                            new_bind[var] = val
                        if ok:
                            sols.append(new_bind)
                elif result:
                    sols = [bindings]
                else:
                    sols = []

            # Για κάθε λύση του πρώτου goal, προχωράμε αναδρομικά
            solutions = []
            for sol in sols:
                solutions.extend(solve_goals(remaining, sol))
            return solutions

        sols = solve_goals(goals, {})
        return sols if sols else False



    def execute_not(self, not_instruction):
        if not isinstance(not_instruction, NotInstruction):
            raise ValueError("execute_not accepts only NotInstruction objects")
        
        goal_with_bindings = self._apply_bindings(fresh_copy(not_instruction.goal), {})
        result = self.execute_goal(goal_with_bindings)
        
        if not result:
            return [ {} ]
        else:
            return []

    def _reset_unbound_variables(self, goal):
        if isinstance(goal, Predicate):
            for term in goal.terms:
                if isinstance(term, Variable):
                    term.value = None  # Επαναφορά της μεταβλητής


    def _apply_bindings(self, goal, bindings):
        if isinstance(goal, Predicate):
            new_terms = []
            for term in goal.terms:
                if isinstance(term, Variable) and term.name in bindings:
                    bound = bindings[term.name]
                    # propagate Variables (so Y stays a variable),
                    # propagate full Term or Constant objects,
                    # only wrap "bare" Python things into Constant(...)
                    if isinstance(bound, (Variable, Constant, Term)):
                        new_terms.append(bound)
                    else:
                        new_terms.append(Constant(bound))
                else:
                    new_terms.append(term)
            return Predicate(goal.name, new_terms)

        elif isinstance(goal, NotInstruction):
            return NotInstruction(self._apply_bindings(goal.goal, bindings))
        elif isinstance(goal, AndInstruction):
            return AndInstruction([self._apply_bindings(g, bindings) for g in goal.goals])
        elif isinstance(goal, OrInstruction):
            return OrInstruction([self._apply_bindings(g, bindings) for g in goal.goals])
        else:
            return goal






        
               
        

       

    def _has_unbound_variables(self, goal):
        if isinstance(goal, Predicate):
            return any(isinstance(term, Variable) and term.value is None for term in goal.terms)
        elif isinstance(goal, NotInstruction):
            return self._has_unbound_variables(goal.goal)
        return False

    def _reset_query_vars(self, query):
        for var in self._collect_variables(query):
            var.value = None
    



        

    def execute_or(self, or_instruction):
        solutions = []
        for goal in or_instruction.goals:
            if isinstance(goal, Predicate):
                res = self.execute_query(goal)
            elif isinstance(goal, NotInstruction):
                res = self.execute_not(goal)
            elif isinstance(goal, AndInstruction):
                res = self.execute_and(goal)
            elif isinstance(goal, OrInstruction):
                res = self.execute_or(goal)
            else:
                res = False

            if isinstance(res, list):
                solutions.extend(res)
            else:
                if res:
                    solutions.append({})
        return solutions if solutions else []

    def _execute_with_bindings(self, goal, bindings):
        
        if isinstance(goal, Predicate):
            # Apply bindings to variables in the goal
            bound_terms = []
            for term in goal.terms:
                if isinstance(term, Variable) and term.name in bindings:
                    bound_terms.append(Constant(bindings[term.name]))
                else:
                    bound_terms.append(term)
            bound_goal = Predicate(goal.name, bound_terms)
            return self.match_fact(bound_goal)
        elif isinstance(goal, NotInstruction):
            return self.execute_not(goal)
        return False


    def find_all_rule_solutions(self, query):
        solutions = []
        for orig_head, orig_body in self.rules:
            # 1) rename/head‐and‐body so no clashes
            env = {}
            fresh_head = fresh_copy(orig_head, env)
            fresh_body = [fresh_copy(g, env) for g in orig_body]

            # 2) unify the user‐query against that fresh head
            head_bindings = {}
            if not self._unify_with_bindings(query, fresh_head, head_bindings):
                continue

            # 3) solve the body as one big conjunction
            body_instr = AndInstruction(fresh_body)
            body_solutions = self.execute_and(body_instr)
            if not body_solutions:
                continue

            # 4) for each body solution, extract the _final_ value
            #    of each query variable from the body env
            for body_env in body_solutions:
                result = {}
                for qvar, rule_var in head_bindings.items():
                    # rule_var is the fresh_head Variable; its .name matches
                    # the body‐env keys in body_env
                    val = body_env.get(rule_var.name)
                    # if the body never bound it (e.g. facts with no body),
                    # maybe it was a constant in the head:
                    if val is None and isinstance(rule_var, Constant):
                        val = rule_var.value
                    result[qvar.name] = val
                solutions.append(result)

        return solutions




        
    def find_all_solutions(self, query):
        display_vars =self._collect_variables(query)
        solutions = []
        if isinstance(query, Predicate):
            for fact in self.knowledge_base:
                if query.name == fact.name and len(query.terms) == len(fact.terms):
                    bindings = {}
                    if self._unify_with_bindings(query, fact, bindings):
                        if display_vars:
                            solution = {}
                            for var in display_vars:
                                if var in bindings:
                                    value = bindings[var]
                                    if isinstance(value, Constant):
                                        solution[var.name] = value.value
                                    else:
                                        solution[var.name] = value
                            solutions.append(solution)
                        else:
                            # Για ανώνυμες μεταβλητές, προσθέτουμε μια dummy λύση
                            solutions.append({})
        return solutions





    def _collect_variables(self, query):
        vars_ = []
        def collect(term):
            if isinstance(term, Variable) and term not in vars_:
                vars_.append(term)
            elif isinstance(term, Predicate):
                for sub in term.terms:
                    collect(sub)
            elif isinstance(term, Term):
                for sub in term.args:
                    collect(sub)
        if isinstance(query, Predicate):
            for t in query.terms:
                collect(t)
        elif isinstance(query, NotInstruction):
            collect(query.goal)
        elif isinstance(query, AndInstruction) or isinstance(query, OrInstruction):
            for g in query.goals:
                for v in self._collect_variables(g):
                    if v not in vars_:
                        vars_.append(v)
        return vars_



    def _unify_with_bindings(self, query_term, fact_term, bindings):
        """Unifies terms while tracking variable bindings"""
        if isinstance(query_term, Variable):
            if query_term in bindings:
                return bindings[query_term].unify(fact_term)
            bindings[query_term] = fact_term
            return True
        elif isinstance(query_term, Predicate) and isinstance(fact_term, Predicate):
            if query_term.name != fact_term.name or len(query_term.terms) != len(fact_term.terms):
                return False
            return all(self._unify_with_bindings(q, f, bindings) 
                      for q, f in zip(query_term.terms, fact_term.terms))
        elif isinstance(query_term, Constant) and isinstance(fact_term, Constant):
            return query_term.value == fact_term.value
        return False
        
    
    
    def save_state(self):
        """Αποθηκεύει την τρέχουσα κατάσταση (μεταβλητές και registers) στη στοίβα."""
        state = {
            "registers": self.registers.copy(),
            "call_frames": self.call_frames.copy(),
            "call_stack": self.call_stack.copy(),
            "heap": self.heap.copy(),
            "choice_points": self.choice_points.copy(),
            "frame_pointer": self.frame_pointer,
            "index": self.index,  # Αποθήκευση του δείκτη index
        }
        self.state_stack.append(state)
        print("save_state: State saved.")

    def restore_state(self):
        """
        Επαναφέρει την τελευταία αποθηκευμένη κατάσταση από τη στοίβα, αν υπάρχει.
        """
        if self.state_stack:
            state = self.state_stack.pop()
            self.registers = state["registers"]
            self.call_frames = state["call_frames"]
            self.call_stack = state["call_stack"]
            self.heap = state["heap"]
            self.choice_points = state["choice_points"]
            self.frame_pointer = state["frame_pointer"]
            self.index = state["index"]  # Επαναφορά του δείκτ
            
            print("restore_state: State restored.")
        else:
            print("restore_state: No state to restore.")

    def add_choice_point(self, instruction_index, alternatives):
        """
        Προσθέτει ένα νέο σημείο επιλογής.
        """
        choice_point = {
            "instruction_index": instruction_index,
            "state_stack": self.state_stack.copy(),
            "call_stack": self.call_stack.copy(),
            "call_frames": self.call_frames.copy(),
            "heap": self.heap.copy(),
            "choice_points": self.choice_points.copy(),
            "frame_pointer": self.frame_pointer,
            "index": self.index,  # Αποθήκευση του δείκτη
            "alternatives": alternatives,  # Λίστα με τις εναλλακτικές διαδρομές εκτέλεσης
            "current_alt": 0 
        }
        
        self.choice_points.append(choice_point)
        print(f"add_choice_point: Added at instruction {instruction_index} with {len(alternatives)} alternatives.")

    def backtrack(self):
        """
        Εκτελεί backtracking, επαναφέρει την τελευταία αποθηκευμένη κατάσταση.
        """
        if self.choice_points:
            choice_point = self.choice_points[-1]
            current_alt = choice_point["current_alt"]
            alternatives = choice_point["alternatives"]
            
            if current_alt < len(alternatives):
                # Επαναφορά στην επόμενη εναλλακτική του `choice point`
                self.restore_state()
                choice_point["current_alt"] += 1  # Μετάβαση στην επόμενη εναλλακτική
                self.index = alternatives[current_alt]
                print(f"backtrack: Switching to alternative at instruction {self.index}.")
                return self.index
            
            # Αν δεν υπάρχουν άλλες εναλλακτικές, αφαιρούμε το `choice point`
            self.choice_points.pop()
            print("backtrack: No more alternatives, removed choice point.")
        
        print("backtrack: No choice points available.")
        return None
        
    def execute(self, instructions):
        index = 0
        while index < len(instructions):
            instr = instructions[index]
            opcode = instr.opcode
            operands = instr.operands

            if opcode == "put_constant":
                if not self.put_constant(operands[0], operands[1]):
                    index = self.backtrack()
                    if index is None:
                        print("Execution failed with no more choice points.")
                        return False
                    continue
            elif opcode == "put_variable":
                self.put_variable(operands[0], operands[1])
            elif opcode == "get_constant":
                if not self.get_constant(operands[0], operands[1]):
                    index = self.backtrack()
                    if index is None:
                        print("Execution failed with no more choice points.")
                        return False
                    continue
            elif opcode == "get_variable":
                if not self.get_variable(operands[0], operands[1]):
                    index = self.backtrack()
                    if index is None:
                        print("Execution failed with no more choice points.")
                        return False
                    continue
            elif opcode == "call":
                # Προσθέτει το τρέχον index ως σημείο επιλογής πριν από την κλήση
                self.add_choice_point(index, [alt_index for alt_index in range(index + 1, len(instructions))])
                self.call(operands[0], operands[1])
                self.call_stack.append(index)
            elif opcode == "get_structure":
                if not self.get_structure(operands[0], operands[1]):
                    index = self.backtrack()
                    if index is None:
                        print("Execution failed with no more choice points.")
                        return False 
                    continue
            elif opcode == "set_variable":
                self.set_variable(operands[0], operands[1])
            elif opcode == "set_value":
                self.set_value(operands[0], operands[1])
            elif opcode == "unify_constant":
                if not self.unify_constant(operands[0], operands[1]):
                    index = self.backtrack()
                    if index is None:
                        print("Execution failed with no more choice points.")
                        return False
                    continue
            elif opcode == "unify_local_value":
                if not self.unify_local_value(operands[0], operands[1]):
                    index = self.backtrack()
                    if index is None:
                        print("Execution failed with no more choice points.")
                        return False
                    continue
            elif opcode == "not":
                if not self.execute_not(operands[0]):
                    index = self.backtrack()
                    if index is None:
                        print("Execution failed with no more choice points.")
                        return False
                    continue
            elif opcode == "proceed":
                print("Proceeding with execution.")
                return True
            elif opcode == "allocate":
                self.allocate(operands[0])  # Η εντολή allocate
            elif opcode == "deallocate":
                self.deallocate()  # Η εντολή deallocate
            elif opcode == "execute_body":
                body_index = 0
                success = True
                while body_index < len(operands[0]) and success:
                    predicate = operands[0][body_index]
                    self.add_choice_point(index)
                    success = self.execute_predicate(predicate)
                    if not success:
                        index = self.backtrack()
                        if index is None:
                            print("Execution failed in body with no more choice points.")
                            return False
                        body_index -= 1
                    body_index += 1
                if not success:
                    return False

            index += 1
        return False  # Δεν βρέθηκε λύση

    def allocate(self, size):
        """
        Η εντολή allocate δημιουργεί ένα νέο πλαίσιο κλήσης, το οποίο αποθηκεύει
        τις μεταβλητές στο call_frame και τα προσθέτει στην στοίβα.
        """
        self.call_frames.append({'frame': {}, 'size': size, 'previous_fp': self.frame_pointer})  # Αποθήκευση του πλαισίου
        self.frame_pointer = len(self.call_frames) - 1  # Ενημέρωση του frame_pointer στο νέο πλαίσιο
        print(f"allocate: Allocated frame of size {size}. Frame pointer now at {self.frame_pointer}")

    def deallocate(self):
        """
        Η εντολή deallocate αφαιρεί το τελευταίο πλαίσιο κλήσης από τη στοίβα.
        """
        if self.call_frames and self.frame_pointer is not None:
            current_frame = self.call_frames.pop()
            self.frame_pointer = current_frame['previous_fp']  # Επαναφορά του frame_pointer στο προηγούμενο πλαίσιο
            print(f"deallocate: Deallocated frame. Frame pointer now at {self.frame_pointer}")
        else:
            print("deallocate: No frame to deallocate.")

    def execute_predicate(self, predicate):
        """Eκτελεί το κατηγόρημα και ελέγχει αν υπάρχει επιτυχία"""
        instructions = []
        self.compile_predicate(predicate, instructions)
        # Προσθέτουμε σημείο επιλογής για το κατηγόρημα
        self.add_choice_point(self.index)
        
        # Εκτέλεση των εντολών για το συγκεκριμένο κατηγόρημα
        success = self.execute(instructions)
        
        # Αν αποτύχει, γίνεται backtracking
        if not success:
            # Επαναφορά στην προηγούμενη κατάσταση
            index = self.backtrack()
            if index is None:
                print("Execution failed with no more choice points.")
                return False
            return self.execute(instructions)  # Προσπάθεια επανεκτέλεσης με την επαναφορά κατάστασης
        return True

    def get_variable(self, var_name, register):
        var = Variable(var_name)
        if register in self.registers:
            result = self.registers[register].unify(var)
            if not result:
                self.restore_state()  # Επαναφορά κατάστασης αν αποτύχει η ενοποίηση
                print(f"get_variable failed: {var_name} != {register}")
                return False
        else:
            self.registers[register] = var
        print(f"get_variable: {var_name} == {register}")
        return True

    def put_constant(self, value, register):
        const = Constant(value)
        if register in self.registers:
            result = self.registers[register].unify(const)
            if not result:
                self.restore_state()  # Αποκατάσταση κατάστασης αν αποτύχει η ενοποίηση
                print(f"put_constant failed: {value} -> {register}")
                return False
        else:
            self.registers[register] = const
        print(f"put_constant: {value} -> {register}")
        return True
    
    def put_variable(self, var_name, register):
        var = Variable(var_name)
        if register in self.registers:
            result = self.registers[register].unify(var)
            if not result:
                self.restore_state()  # Αποκατάσταση κατάστασης αν αποτύχει η ενοποίηση
                print(f"put_variable failed: {var_name} -> {register}")
                return False
        else:
            self.registers[register] = var
        print(f"put_variable: {var_name} -> {register}")
        return True

    


   

# Load the facts into the knowledge base
# Create an instance of WAMExecutor
executor = WAMExecutor()







# Συνάρτηση εκτέλεσης ερωτημάτων
def run_query(query):
        # ─── Prolog‐style arithmetic is/2 ───

        # ─── Prolog‐style numeric equality =:= ───
    if isinstance(query, Predicate) and query.name == "=:=":
        left, right = query.terms

        # 1) Αν υπάρχει αδέσμευτη μεταβλητή → instantiation error
        for term in (left, right):
            if isinstance(term, Variable) and term.value is None:
                return "ERROR: Arguments are not sufficiently instantiated"

        # 2) Πρέπει να είναι αριθμητικές εκφράσεις
        arith_pat = re.compile(r'^[0-9+\-*/\s()]+$')
        if not (isinstance(left, Constant) and isinstance(left.value, str) and arith_pat.fullmatch(left.value)
                and isinstance(right, Constant) and isinstance(right.value, str) and arith_pat.fullmatch(right.value)):
            return "false"

        # 3) Ασφαλής υπολογισμός
        try:
            v1 = eval(left.value, {"__builtins__": None}, {})
            v2 = eval(right.value, {"__builtins__": None}, {})
        except Exception:
            return "false"

        # 4) Σύγκριση
        return "true" if v1 == v2 else "false"
    # ───────────────────────────────────────────

    if isinstance(query, Predicate) and query.name == "is":
        left, right = query.terms

        # 1) RHS must be an integer expression (digits, +-*/ and parentheses)
        arith_pat = re.compile(r'^[0-9+\-*/\s()]+$')
        if not (isinstance(right, Constant)
                and isinstance(right.value, str)
                and arith_pat.fullmatch(right.value)):
            return "false"

        # 2) Evaluate it safely
        try:
            val = eval(right.value, {"__builtins__": None}, {})
        except Exception:
            return "false"

        # Wrap the result as a Prolog‐Constant
        num_const = Constant(str(val))

        # 3a) If LHS is a variable → bind it once, or check existing binding
        if isinstance(left, Variable):
            if left.value is None:
                left.value = num_const
                return f"{left.name} = {val}"
            else:
                # already bound earlier → succeed only if same number
                try:
                    return "true" if int(left.value.value) == val else "false"
                except:
                    return "false"

        # 3b) If LHS is already a constant → compare numerically
        elif isinstance(left, Constant):
            try:
                return "true" if int(left.value) == val else "false"
            except:
                return "false"

        # anything else (e.g. compound term) fails
        else:
            return "false"
    # ───────────────────────────────────────




    # --- new case for Prolog‐style "=" ---
    if isinstance(query, Predicate) and query.name == "=":
        left, right = query.terms

        # simple recursive unify
        def unify(a, b):
            # variable on left?
            if isinstance(a, Variable):
                a.value = b
                return True
            # variable on right?
            if isinstance(b, Variable):
                b.value = a
                return True
            # two constants?
            if isinstance(a, Constant) and isinstance(b, Constant):
                return a.value == b.value
            # two compound terms?
            if isinstance(a, Term) and isinstance(b, Term):
                if a.functor != b.functor or len(a.args) != len(b.args):
                    return False
                return all(unify(x, y) for x, y in zip(a.args, b.args))
            return False

        success = unify(left, right)
        if not success:
            return "false"
        # collect any named variables to print
        vars_ = [v for v in executor._collect_variables(query) if v.name != '_']
        if not vars_:
            return "true"
        # format each binding as 'X = hello' or 'X = 4+1'
        out = []
        for v in vars_:
            val = v.value
            if isinstance(val, Constant):
                out.append(f"{v.name} = {val.value}")
            else:
                out.append(f"{v.name} = {val}")
        return "\n".join(out)



    # Μέσα στη συνάρτηση run_query, αντικαθιστούμε πλήρως την περίπτωση
# “AndInstruction με 2 goals” ως εξής:

    if isinstance(query, AndInstruction) and len(query.goals) == 2:
        or_branch, cond = query.goals

        # Αν έχουμε (link(s,X,D); link(X,s,D)), D<100 κ.λπ.
        if (isinstance(or_branch, OrInstruction)
            and isinstance(cond, Predicate)
            and cond.name in ("<", ">", "=<", ">=")):

            # Εκτελούμε πρώτα το OR για να πάρουμε όλες τις λύσεις
            raw_sols = executor.execute_or(or_branch)
            filtered = []
            L, R = cond.terms  # L == D, R == Constant("100")

            for bind in raw_sols:
                # Αριστερό όρισμα (π.χ. D)
                if isinstance(L, Variable):
                    dv = bind.get(L.name)
                elif isinstance(L, Constant):
                    dv = L.value
                else:
                    continue

                # Προσπαθούμε να το κάνουμε int
                try:
                    dv = int(dv)
                except:
                    continue

                # Δεξί όρισμα (π.χ. Constant("100"))
                if isinstance(R, Constant):
                    try:
                        rv = int(R.value)
                    except:
                        continue
                else:
                    continue

                # Κάνουμε την αριθμητική σύγκριση
                ok = False
                if   cond.name == "<":  ok = dv < rv
                elif cond.name == ">":  ok = dv > rv
                elif cond.name == "=<": ok = dv <= rv
                elif cond.name == ">=": ok = dv >= rv

                if ok:
                    filtered.append(bind)

            # Αν δεν βρέθηκε καμία λύση
            if not filtered:
                return "false"

            # Μορφοποίηση αποτελεσμάτων σε Prolog-style
            lines = []
            vars_to_show = executor._collect_variables(or_branch)
            for sol in filtered:
                parts = [f"{v.name} = {sol[v.name]}" for v in vars_to_show if v.name in sol]
                lines.append(", ".join(parts))
            lines.append("false")
            return "\n".join(lines)


        

    
    
    
    

    
    if isinstance(query, Predicate) and query.name in ["write", "tab", "nl"] :
        result = executor.execute_query(query)
        return f"{result}\ntrue"
    
    # Νέο case για AndInstruction που αποτελείται μόνο από write predicates
    if isinstance(query, AndInstruction) and all(isinstance(g, Predicate) and g.name in ["write", "tab", "nl"] for g in query.goals):
        output = ""
        for goal in query.goals:
            res = executor.execute_query(goal)
            if res == "false":
                return "false"
            output += res
        return f"{output}\ntrue"
    
    
    if isinstance(query, AndInstruction) and len(query.goals) == 3:
        first_goal, second_goal, third_goal = query.goals

         

        def goal_term_count(goal):
            return len(goal.terms) if isinstance(goal, Predicate) else None

        if goal_term_count(first_goal) == 2 and goal_term_count(second_goal) == 2:

            
            # Έλεγχος αν ο κοινός όρος είναι ανώνυμος ("_")
            if (isinstance(first_goal.terms[0], Variable) and first_goal.terms[0].name == "_" and
                isinstance(second_goal.terms[0], Variable) and second_goal.terms[0].name == "_"):
                # Εξαγωγή των ονομάτων των παιδικών μεταβλητών από τα δύο predicates
                var1_name = first_goal.terms[1].name if isinstance(first_goal.terms[1], Variable) else None
                var2_name = second_goal.terms[1].name if isinstance(second_goal.terms[1], Variable) else None

                sol1 = executor.execute_query(first_goal)
                sol2 = executor.execute_query(second_goal)
                result_lines = []
                for s1 in sol1:
                    for s2 in sol2:
                        # Χρησιμοποιούμε τα ονόματα που εξήχθησαν (π.χ. F και G)
                        if s1.get(var1_name) != s2.get(var2_name):
                            combined = {}
                            combined.update(s1)
                            combined.update(s2)
                            third_bound = executor._apply_bindings(fresh_copy(third_goal), combined)
                            if isinstance(third_bound, Predicate):
                                third_result = executor.execute_query(third_bound)
                            elif isinstance(third_bound, NotInstruction):
                                third_result = executor.execute_not(third_bound)
                            else:
                                third_result = False
                            if third_result:
                                result_lines.append(f"{var1_name} = {s1.get(var1_name)}, {var2_name} = {s2.get(var2_name)}")
                if not result_lines:
                    return "false"
                result_lines.append("false")
                return "\n".join(result_lines)
            else:
                # Επεξεργασία για την περίπτωση όπου ο πρώτος όρος είναι κοινός (π.χ. _X)
                def rename_vars(goal):
                    if isinstance(goal, Predicate):
                        new_terms = []
                        for term in goal.terms:
                            if isinstance(term, Variable) and term.name.startswith('_'):
                                new_terms.append(Variable(term.name.lstrip('_')))
                            else:
                                new_terms.append(term)
                        return Predicate(goal.name, new_terms)
                    return goal

                renamed_first_goal = rename_vars(first_goal)
                renamed_second_goal = rename_vars(second_goal)
                combined = AndInstruction([renamed_first_goal, renamed_second_goal])
                all_solutions = executor.execute_and(combined)
                if not all_solutions:
                    return "false"

                # Εξαγωγή του ονόματος του κοινού όρου και των 
                common_var_name = None
                var1_name = None
                var2_name = None
                if isinstance(first_goal, Predicate) and len(first_goal.terms) >= 2:
                    if isinstance(first_goal.terms[0], Variable):
                        common_var_name = first_goal.terms[0].name
                    if isinstance(first_goal.terms[1], Variable):
                        var1_name = first_goal.terms[1].name
                if isinstance(second_goal, Predicate) and len(second_goal.terms) >= 2:
                    if isinstance(second_goal.terms[0], Variable):
                        if common_var_name is None:
                            common_var_name = second_goal.terms[0].name
                        elif common_var_name != second_goal.terms[0].name:
                            common_var_name = None
                    if isinstance(second_goal.terms[1], Variable):
                        var2_name = second_goal.terms[1].name

                

                groups = {}
                if common_var_name is not None:
                    for sol in all_solutions:
                        c_val = sol.get(common_var_name)
                        groups.setdefault(c_val, []).append(sol)
                else:
                    groups[None] = all_solutions

                result_lines = []
                for group, sols in groups.items():
                    for sol in sols:
                        if var1_name is not None and var2_name is not None:
                            if sol.get(var1_name) != sol.get(var2_name):
                                third_bound = executor._apply_bindings(fresh_copy(third_goal), sol)
                                if isinstance(third_bound, Predicate):
                                    third_result = executor.execute_query(third_bound)
                                elif isinstance(third_bound, NotInstruction):
                                    third_result = executor.execute_not(third_bound)
                                else:
                                    third_result = False

                                if third_result:
                                    if not common_var_name.startswith('_'):
                                        result_lines.append(f"{common_var_name} = {sol.get(common_var_name)}, {var1_name} = {sol.get(var1_name)}, {var2_name} = {sol.get(var2_name)}")
                                    else:
                                        result_lines.append(f"{var1_name} = {sol.get(var1_name)}, {var2_name} = {sol.get(var2_name)}")
                if not result_lines:
                    return "false"
                result_lines.append("false")
                return "\n".join(result_lines)

    # Επεξεργασία για AndInstruction με 2 goals (διατηρείται ο υπάρχων κώδικας)
    if isinstance(query, AndInstruction) and len(query.goals) == 2:
        def goal_term_count(goal):
            if isinstance(goal, Predicate):
                return len(goal.terms)
            elif isinstance(goal, NotInstruction) and isinstance(goal.goal, Predicate):
                return len(goal.goal.terms)
            return None

        left_goal = query.goals[0]
        right_goal = query.goals[1]
        left_count = goal_term_count(left_goal)
        right_count = goal_term_count(right_goal)

        if left_count == 2 and right_count == 2:
            left_results = (
                executor.execute_query(left_goal)
                if not isinstance(left_goal, NotInstruction)
                else executor.execute_not(left_goal)
            )
            if left_results == "false" or (isinstance(left_results, list) and not left_results):
                return "false"
            right_results = (
                executor.execute_query(right_goal)
                if not isinstance(right_goal, NotInstruction)
                else executor.execute_not(right_goal)
            )
            if right_results in ("false", False) or (isinstance(right_results, list) and not right_results):
                return "\n".join(["false" for _ in left_results])
            left_pred = left_goal.goal if isinstance(left_goal, NotInstruction) else left_goal
            right_pred = right_goal.goal if isinstance(right_goal, NotInstruction) else right_goal
            display_vars = executor._collect_variables(query)
            left_var_names = set(term.name for term in left_pred.terms if isinstance(term, Variable))
            right_var_names = set(term.name for term in right_pred.terms if isinstance(term, Variable))
            common_var_names = left_var_names.intersection(right_var_names)

            output_lines = []
            #failed_branch = False
            # Για κάθε λύση του αριστερού στόχου, ελέγχουμε αν υπάρχει τουλάχιστον μία join λύση
            for i, left_sol in enumerate(left_results):
                found = False
                for right_sol in right_results:
                    join_possible = True
                    for var in common_var_names:
                        if var not in left_sol or var not in right_sol or left_sol[var] != right_sol[var]:
                            join_possible = False
                            break
                    if join_possible:
                        found = True
                        merged = left_sol.copy()
                        merged.update(right_sol)
                        parts = []
                        for var in display_vars:
                            if var.name in merged and not var.name.startswith("_"):
                                parts.append(f"{var.name} = {merged[var.name]}")
                        output_lines.append(", ".join(parts))
                if not found and i==len(left_results) - 1:
                    #failed_branch = True
                    output_lines.append("false")
            return "\n".join(output_lines)
        elif left_count == 1 and right_count == 1:
            left_str = run_query(left_goal).strip()
            if left_str == "false":
                return "false"
            right_str = run_query(right_goal).strip()
            if left_str in ["true", "false"] and right_str in ["true", "false"]:
                if left_str == "true" and right_str == "true":
                    return "true"
                else:
                    return "false"
            else:
                return f"{left_str}\n{right_str}"


    if isinstance(query, Predicate):
        result = executor.execute_query(query)
    elif isinstance(query, OrInstruction):
        results = [run_query(sub_goal) for sub_goal in query.goals]
        if len(results) == 2 and results[0] == "false" and results[1] != "false":
            return results[1]
        else:
            return "\n".join(results)
    elif isinstance(query, NotInstruction):
        result = executor.execute_not(query)
    else:
        result = False

    display_vars = [var for var in executor._collect_variables(query) if var.name != "_"]

    if isinstance(result, list):
        if not result:
            return "false"
        if not display_vars:
            return "  true"
        has_bindings = any(
            any(var.name in sol for var in display_vars)
            for sol in result
        )
        if not has_bindings:
            return "true"
        formatted_solutions = []
        for solution in result:
            if solution is False:  # Ελέγχουμε αν πρόκειται για branch αποτυχίας
                formatted_solutions.append("false")
            else:
                solution_str = ", ".join(
                    f"{var.name} = {solution[var.name]}"
                    for var in display_vars
                    if var.name in solution
                )
            if solution_str:
                formatted_solutions.append(solution_str)
        return "\n".join(formatted_solutions) if formatted_solutions else "true"
    return "true" if result else "false"

    






    


def load_queries_from_file(filename):
    """Reads queries from a file (lines starting with "?-") and returns them as objects."""
    queries = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("?-"):
                # Remove the "?-" and the trailing period if present.
                query_str = line[2:].strip()
                if query_str.endswith('.'):
                    
                    query_str = query_str[:-1].strip()
                    query = parse_query(query_str)
                    queries.append(query)

    if len(queries)>0:
        print ("Here are the queries that are being executed:")

    for query in queries:
    

        print(f"?- {query}")
        print(run_query(query))  # `run_query` remains unchanged/'η 'run_query' δεν αλλάζει
                    
    return queries

















