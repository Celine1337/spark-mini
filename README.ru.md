<p align="center"><img src="docs/style/hero.png" alt="Spark Mini — локальный ассистент для кода и текста" width="100%"></p>

<p align="center">
  <a href="README.md"><img src="docs/style/pill-en.png" alt="README EN" height="42"></a>
  <a href="https://github.com/Celine1337/spark-mini/releases"><img src="docs/style/pill-release.png" alt="Releases" height="42"></a>
  <a href="https://github.com/Celine1337/spark-mini/actions/workflows/ci.yml"><img src="docs/style/pill-ci.png" alt="CI" height="42"></a>
  <a href="LICENSE"><img src="docs/style/pill-license.png" alt="MIT" height="42"></a>
</p>

<img src="docs/style/sec-about-ru.png" alt="О проекте" width="100%">

**Spark Mini** — локальный ассистент для кода и текста: персона, CLI на
стандартной библиотеке, поиск контекста по вашему репозиторию и опциональный
рецепт LoRA-дообучения — поверх **открытой базовой модели** (по умолчанию
[Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)).

Чем это **не** является: фундаментальной моделью, обученной с нуля. В репозитории
нет обученных весов, и никто не обещает «идеальную» модель. Интеллект даёт
открытая базовая модель, на которую вы указываете; Spark Mini добавляет персону,
промпты, контекст репозитория, eval-харнесс и инструментарий дообучения.

Сам CLI — только стандартная библиотека (Python 3.9+). Инференс делегируется
локальному [Ollama](https://ollama.com) или любому OpenAI-совместимому серверу
(llama.cpp, vLLM, LM Studio). Встроенный бэкенд `mock` выдаёт детерминированные
заглушки, поэтому демо, тесты и CI работают офлайн **без загруженной модели** —
все скриншоты ниже помечены `mock` именно поэтому.

<img src="docs/style/sec-features-ru.png" alt="Возможности" width="100%">

- Пять режимов: `ask`, интерактивный `chat`, `code` (write / fix / explain /
  review / test одного файла), `repo` (ответ с найденным контекстом репозитория),
  `write` (текст в тоне plain / concise / friendly / formal).
- RAG-lite контекст: TF-IDF оценка терминов по исходникам, top-k фрагментов в
  промпт — без эмбеддингов и векторных БД, только stdlib.
- Три бэкенда за одним интерфейсом: `ollama`, `openai` (совместимый), `mock`.
  Переключение `--backend` или `SPARK_BACKEND`; адрес `--base-url`.
- `--json` машинный вывод (persona, backend, model, response, context);
  `--temperature`, `--max-tokens`, `--timeout`; `--no-color` / `NO_COLOR`.
- Опциональный GPU-пайплайн в `finetune/`: подготовка chat-датасета, обучение
  LoRA / QLoRA адаптера, слияние и оценка маленьким офлайн eval-харнессом.
- Честные ошибки: недоступный бэкенд → exit `1` с понятным сообщением; нет
  файла / каталога → exit `2`. Без тихих фолбэков и выдуманных ответов.

<img src="docs/style/sec-checks-ru.png" alt="Режимы и пайплайн" width="100%">

| Команда | Что делает |
|---------|------------|
| `spark.py ask "…"` | одиночный вопрос |
| `spark.py chat` | интерактивная сессия (`/exit` — выход) |
| `spark.py code <task> --file F` | write / fix / explain / review / test одного файла |
| `spark.py repo "…" --root DIR` | ответ по top-k найденным фрагментам репозитория |
| `spark.py write "…" --style S` | текст в выбранном тоне |

Пайплайн дообучения (нужен ваш GPU; веса **не** входят в репозиторий):

| Шаг | Скрипт | Примечания |
|-----|--------|------------|
| 1 | `finetune/prepare_data.py` | stdlib; `--jsonl` пары или `--code DIR` self-supervised продолжение; дедупликация + train/val |
| 2 | `finetune/train_lora.py` | LoRA / QLoRA (PEFT) на базовой модели; защищённые импорты, exit `2` с подсказкой установки без зависимостей/GPU |
| 3 | `finetune/merge_lora.py` | слияние адаптера с базой; конвертация в GGUF через llama.cpp для лёгкого локального рантайма |
| 4 | `finetune/eval.py` | stdlib-оценка: код должен парситься, текст — содержать ключевое слово; `--backend mock` офлайн |

<img src="docs/style/sec-stack-ru.png" alt="Технологии" width="100%">

<img src="docs/style/chips.png" alt="Python 3.9+, stdlib CLI, Ollama / OpenAI API; LoRA / QLoRA (PEFT), unittest, Pillow" width="100%">

<img src="docs/style/sec-quick-ru.png" alt="Быстрый старт" width="100%">

Офлайн, без модели (это и гоняет CI):

```console
$ python spark.py ask "hi" --backend mock
$ python spark.py code explain --file spark.py --backend mock
$ python spark.py repo "how does context selection work?" --backend mock
$ python finetune/eval.py --backend mock
```

С реальной локальной моделью (Ollama):

```console
$ ollama pull qwen2.5-coder:1.5b
$ python spark.py code review --file app.py
$ python spark.py repo "where is auth handled?" --root .
$ python spark.py write "release notes for v1.1" --style concise
```

OpenAI-совместимый сервер (llama.cpp / vLLM / LM Studio):

```console
$ python spark.py ask "explain this stack trace" --backend openai --base-url http://localhost:8080
```

Дообучение на вашем GPU:

```console
$ pip install torch transformers peft datasets accelerate bitsandbytes
$ python finetune/prepare_data.py --code ./src
$ python finetune/train_lora.py --data data
$ python finetune/merge_lora.py --adapter spark-lora --out spark-merged
$ python finetune/eval.py --backend ollama --model spark-merged
```

Коды выхода: `0` — успех, `1` — бэкенд недоступен / eval провален, `2` — плохой
путь, нет данных или нет GPU-зависимостей.

<img src="docs/style/sec-shots-ru.png" alt="Скриншоты" width="100%">

Оба скриншота используют офлайн-бэкенд `mock` — заглушки, модель не загружена.

![режим code](docs/code.png)

![режим repo с найденным контекстом](docs/repo.png)

<img src="docs/style/sec-tests-ru.png" alt="Тесты" width="100%">

```console
$ python -m unittest discover -s tests -v
```

32 тестов покрывают сборщики промптов, все три бэкенда (включая локальный
фейковый HTTP-сервер, проверяющий пути и тела запросов), TF-IDF селектор
контекста, коды выхода CLI, подготовку данных, eval-оценку и корректную
деградацию GPU-скриптов без зависимостей. Картинки воспроизводимы:

```console
$ pip install pillow
$ python tools/make_shots.py      # docs/code.png, docs/repo.png (mock backend)
$ python tools/make_style.py tools/style.json docs/style
```

<img src="docs/style/stats.png" alt="32 тестов, 3 ОС в CI, v1.0.0" width="100%">

[MIT](LICENSE) © Celine1337