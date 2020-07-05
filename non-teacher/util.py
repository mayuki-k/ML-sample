def show_score(f, x, y, name = ''):
    print(f'{name} Accuracy: {f.score(x, y):.3f}')