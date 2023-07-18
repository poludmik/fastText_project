def expand_text(text):
    if text == "":
        return [""]
    tree = And(text)
    return tree.generate()


def find_top_level_ors(text):
    or_positions = []
    level = 0
    for i, token in enumerate(text):
        if token == "(":
            level += 1
        if token == ")":
            level -= 1
        if token == "|" and level == 0:
            or_positions.append(i)
        assert level >= 0, "Brackets mismatch in text %s" % text
    assert level == 0, "Brackets mismatch in text %s" % text
    return or_positions


def divide_ors(text, or_positions):
    parts = []
    start = 0
    for position in or_positions:
        parts.append(text[start:position])
        start = position + 1
    parts.append(text[start:])
    return parts


def find_top_level_brackets(text):
    brackets_positions = []
    level = 0
    for i, token in enumerate(text):
        if token == "(":
            if level == 0:
                brackets_positions.append(i)
            level += 1
        if token == ")":
            level -= 1
            if level == 0:
                brackets_positions.append(i)
        assert level >= 0, "Brackets mismatch in text %s" % text

    assert level == 0, "Brackets mismatch in text %s" % text
    return brackets_positions


def divide_brackets(text, brackets_positions):
    parts = []
    start = 0
    for position in brackets_positions:
        parts.append(text[start:position])
        start = position + 1
    parts.append(text[start:])
    parts = [x for x in parts if x != ""]
    return parts


class Or:
    def __init__(self, text):
        self.childs = []
        positions = find_top_level_ors(text)
        parts = divide_ors(text, positions)
        for part in parts:
            if ("(" not in part or ")" not in part) and "|" not in part:
                self.childs.append(part)
            else:
                node = And(part)
                self.childs.append(node)

    def generate(self):
        to_return = []
        for child in self.childs:
            if isinstance(child, str):
                to_return += [child]
            else:
                to_return += child.generate()
        return to_return


class And:
    def __init__(self, text):
        self.childs = []
        positions = find_top_level_brackets(text)
        parts = divide_brackets(text, positions)
        for part in parts:
            if ("(" not in part or ")" not in part) and "|" not in part:
                self.childs.append(part)
            else:
                node = Or(part)
                self.childs.append(node)

    def generate(self):
        parts = []
        for child in self.childs:
            if isinstance(child, str):
                parts.append([child])
            else:
                parts.append(child.generate())
        to_return = self.expand(parts)
        return to_return

    def expand(self, parts):
        return self._expand(parts[1:], parts[0])

    def _expand(self, parts, acc):
        if len(parts) == 0:
            return acc
        new_acc = []
        for part in parts[0]:
            for a in acc:
                new_acc.append(a + part)
        return self._expand(parts[1:], new_acc)

if __name__ == '__main__':
    print(expand_text("Hello (how are you| howdy (|you))? (A|B)"))