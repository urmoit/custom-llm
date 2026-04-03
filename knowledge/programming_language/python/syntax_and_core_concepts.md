# Python Syntax and Core Concepts

## Syntax and Semantics
Python uses indentation to define blocks instead of braces. Recommended indentation is four spaces. Semicolons are allowed but uncommon.

Core statements include:
- `if` / `elif` / `else`
- `for` and `while`
- `try` / `except` / `finally`
- `def`, `class`, `with`, `return`, `yield`, `raise`, `import`
- `match` / `case` (structural pattern matching)

## Core Language Features
- Dynamic typing with strong runtime type behavior.
- Automatic memory management (reference counting plus cycle detection in CPython).
- Duck typing and late binding.
- First-class functions and closures.
- Comprehensions and generator expressions.
- Rich exception handling model.

## Built-in Data Types (Important Set)
- Numeric: `int`, `float`, `complex`, `bool`
- Text/binary: `str`, `bytes`, `bytearray`
- Collections: `list`, `tuple`, `dict`, `set`, `frozenset`, `range`
- Special singletons: `None`, `NotImplemented`, `Ellipsis`

Mutability matters:
- Mutable: `list`, `dict`, `set`, `bytearray`
- Immutable: `int`, `float`, `bool`, `str`, `tuple`, `frozenset`, `bytes`

## Operators and Expression Notes
- Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Matrix multiplication: `@`
- Assignment expression (walrus): `:=`
- Identity vs equality: `is` versus `==`
- Chained comparisons are supported: `a < b < c`

Division behavior in Python 3:
- `/` is true division (float result)
- `//` is floor division
