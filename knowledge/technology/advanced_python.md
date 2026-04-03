# Advanced Python Programming

## Decorators and Context Managers
Decorators wrap functions to add behavior without modifying them. A decorator is a function that takes a function and returns a modified version. The @functools.wraps decorator preserves the wrapped function's metadata. Common uses: logging, timing, authentication, caching, retry logic. Context managers implement __enter__ and __exit__ (or use @contextmanager from contextlib). The `with` statement ensures cleanup even if exceptions occur. Example uses: file handling, database connections, locks, and timers.

## Concurrency and Parallelism
Python has three main concurrency models. Threading: multiple threads in one process; limited by the GIL for CPU-bound tasks but useful for I/O-bound tasks. Multiprocessing: separate processes bypass the GIL; good for CPU-bound tasks. Asyncio: single-threaded cooperative concurrency using async/await; excellent for I/O-bound and network-heavy code. The GIL (Global Interpreter Lock) prevents true parallel execution of Python bytecode in CPython threads. asyncio event loop runs coroutines; `async def` defines coroutines; `await` pauses execution until the awaited coroutine completes.

## Data Classes and Type System
Python 3.7+ dataclasses (@dataclass decorator) auto-generate __init__, __repr__, and __eq__. Field options: default values, field(), post_init. TypedDict for typed dictionaries. Literal types restrict values to specific constants. Union types (X | Y in Python 3.10+). Protocol defines structural subtyping (duck typing with type checking). TypeVar for generic functions. ParamSpec and Concatenate for decorator typing. Runtime type checking via isinstance(); static checking via mypy or pyright.

## Testing Best Practices
pytest is the most popular Python testing framework. Test discovery: files named test_*.py, functions named test_*. Fixtures provide reusable setup/teardown. Parametrize runs the same test with multiple inputs. Mocking (unittest.mock): Mock, MagicMock, patch. Test coverage measured with pytest-cov. TDD (Test-Driven Development) writes tests before implementation. Integration tests test component interactions. Property-based testing (hypothesis library) generates random test cases automatically.

## Performance Optimization
Profile first — don't optimize blind. cProfile and line_profiler identify bottlenecks. Big-O analysis: prefer O(n log n) over O(n²) for large datasets. Use sets for O(1) membership tests instead of lists O(n). NumPy vectorization avoids Python loops for numerical operations. Cython compiles Python to C for speed. Numba JIT-compiles numerical Python. lru_cache (functools) memoizes expensive pure functions. Generator expressions avoid building large lists in memory. slots=True in dataclasses reduces memory by avoiding __dict__.

## Packaging and Distribution
pyproject.toml is the modern standard for project metadata and build configuration. setuptools, flit, and hatch are popular build backends. Virtual environments: venv (built-in), virtualenv, conda. Dependency pinning: requirements.txt (exact pins), pyproject.toml (ranges), pip-compile for reproducible locks. Publishing to PyPI via `pip install build` and `twine upload`. Semantic versioning: MAJOR.MINOR.PATCH. Namespace packages allow splitting a package across multiple directories. __init__.py makes a directory a package; omitting it creates a namespace package.

## Design Patterns in Python
Singleton: use module-level variables (Python modules are singletons by default). Factory: functions or classes that create objects based on parameters. Observer: use callbacks, signals, or the built-in logging system. Strategy: pass functions as arguments (first-class functions make this natural). Command: wrap operations as callable objects. Repository: abstract data access behind an interface. Dependency injection: pass dependencies as constructor arguments rather than hardcoding them. SOLID principles apply: Single responsibility, Open/closed, Liskov substitution, Interface segregation, Dependency inversion.

## Python Web Frameworks
Flask is a lightweight microframework — minimal core, extensions for everything else. FastAPI is modern, async, and auto-generates OpenAPI docs from type hints. Django is a full-featured framework with ORM, admin, auth, and templates. Uvicorn is the ASGI server for FastAPI/Starlette. SQLAlchemy is the dominant Python ORM — supports sync and async. Pydantic v2 is the data validation library underlying FastAPI. REST APIs use HTTP verbs (GET, POST, PUT, DELETE, PATCH). GraphQL is an alternative query language for APIs.
