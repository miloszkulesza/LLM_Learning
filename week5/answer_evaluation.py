import test
from eval import evaluate_retrieval, evaluate_answer

tests = test.load_tests()

print(len(tests))

example = tests[0]
print(example.question)
print(example.category)
print(example.reference_answer)
print(example.keywords)

from collections import Counter
count = Counter([t.category for t in tests])
print(count)

print(evaluate_retrieval(example))

eval, answer, chunks = evaluate_answer(example)
print(eval.feedback)
print(eval.accuracy)
print(eval.completeness)
print(eval.relevance)