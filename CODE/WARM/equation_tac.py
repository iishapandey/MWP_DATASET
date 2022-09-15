class EquationTAC:
    def __init__(self, et, nums):
        self.op_vocab = ['+', '-', '*', '/', '^']
        self.num_vocab = nums
        self.eq_tac = []

        self.initialize(et)

    def initialize_num_vocab(self, et):
        if et is None:
            return
        else:
            if et.value not in self.op_vocab:
                self.num_vocab.append(et.value)
            self.initialize_num_vocab(et.left)
            self.initialize_num_vocab(et.right)

    def initialize(self, et):

        if et is None or et.value not in self.op_vocab:
            return
        if et.left is not None and et.left.value and et.left.left is None and et.left.right is None:
            et.left.value = self.num_vocab.index(float(et.left.value))
        else:
            self.initialize(et.left)
        if et.right is not None and et.right.left is None and et.right.right is None:
            et.right.value = self.num_vocab.index(float(et.right.value))
        else:
            self.initialize(et.right)
        self.eq_tac.append([self.op_vocab.index(et.value), et.left.value, et.right.value])
        self.num_vocab.append(self.execute(et.value, self.num_vocab[et.left.value], self.num_vocab[et.right.value]))
        et.value = len(self.num_vocab) - 1

    def execute(self, op, op1, op2):
        if op == '+':
            return float(op1) + float(op2)
        elif op == '-':
            return float(op1) - float(op2)
        elif op == '*':
            return float(op1) * float(op2)
        elif op == '/':
            return float(op1) / float(op2)
        elif op == '^':
            return float(op1) ** float(op2)

    def print(self):
        print(self.num_vocab)
        print(self.eq_tac)