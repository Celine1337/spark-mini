import json
import os
import tempfile
import unittest

import retrieval as rt


class Tokenizer(unittest.TestCase):
    def test_split_camel(self):
        self.assertEqual(rt.split_identifier("invoiceTotal"), ["invoice", "total"])

    def test_split_snake(self):
        self.assertEqual(rt.split_identifier("invoice_total"), ["invoice", "total"])

    def test_split_acronym(self):
        self.assertEqual(rt.split_identifier("parseHTTPResponse"),
                         ["parse", "http", "response"])

    def test_split_single(self):
        self.assertEqual(rt.split_identifier("total"), ["total"])

    def test_tokenize_subtokens(self):
        toks = rt.tokenize("def invoiceTotal(x):")
        self.assertIn("invoicetotal", toks)
        self.assertIn("invoice", toks)
        self.assertIn("total", toks)

    def test_tokenize_stopwords_removed(self):
        toks = rt.tokenize("the and of is a")
        self.assertEqual(toks, [])

    def test_tokenize_min_length(self):
        toks = rt.tokenize("a I x ab cd")
        self.assertIn("ab", toks)
        self.assertNotIn("x", toks)

    def test_query_terms_min_three(self):
        self.assertEqual(rt.query_terms("ab"), [])
        self.assertEqual(rt.query_terms("ab cd"), [])
        self.assertEqual(rt.query_terms("abc"), ["abc"])

    def test_query_terms_dedup_order(self):
        self.assertEqual(rt.query_terms("invoice invoice total"), ["invoice", "total"])

    def test_query_terms_camel_match(self):
        # camelCase query produces sub-tokens that match snake_case docs
        terms = rt.query_terms("invoiceTotal")
        self.assertIn("invoice", terms)
        self.assertIn("total", terms)


class Understand(unittest.TestCase):
    def test_terms_identifiers_filenames(self):
        u = rt.understand("where is invoice_total in billing.py?")
        self.assertIn("invoice", u["terms"])
        self.assertIn("invoice_total", u["identifiers"])
        self.assertIn("billing.py", u["filenames"])

    def test_windows_path_basename(self):
        u = rt.understand("look at src\\billing.py")
        self.assertIn("billing.py", u["filenames"])

    def test_estimate_tokens(self):
        self.assertEqual(rt.estimate_tokens(""), 1)
        self.assertEqual(rt.estimate_tokens("abcd" * 10), 10)


class Chunking(unittest.TestCase):
    def test_code_def_boundaries_and_symbols(self):
        src = ("import os\n\n"
               "def alpha():\n    return 1\n\n"
               "def beta():\n    return 2\n")
        chunks = rt.chunk_text("m.py", src, ".py")
        names = [c["symbols"] for c in chunks]
        self.assertIn(["alpha"], names)
        self.assertIn(["beta"], names)
        alpha = next(c for c in chunks if c["symbols"] == ["alpha"])
        self.assertIn("return 1", alpha["text"])
        self.assertNotIn("return 2", alpha["text"])

    def test_code_line_ranges_one_based(self):
        src = "def alpha():\n    return 1"
        chunks = rt.chunk_text("m.py", src, ".py")
        self.assertEqual(chunks[0]["start"], 1)
        self.assertEqual(chunks[0]["end"], 2)

    def test_code_class_method_nesting(self):
        src = ("class Foo:\n"
               "    def bar(self):\n        return 1\n"
               "    def baz(self):\n        return 2\n")
        chunks = rt.chunk_text("m.py", src, ".py")
        syms = [s for c in chunks for s in c["symbols"]]
        self.assertIn("Foo", syms)
        self.assertIn("bar", syms)
        self.assertIn("baz", syms)

    def test_code_no_defs_windows(self):
        src = "x = 1\ny = 2\nz = 3\n"
        chunks = rt.chunk_text("m.py", src, ".py")
        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["kind"], "code")

    def test_prose_paragraphs(self):
        para = "word " * 60  # > MIN_PARA_CHARS
        src = para + "\n\n" + para + "\n"
        chunks = rt.chunk_text("d.md", src, ".md")
        self.assertEqual(chunks[0]["kind"], "prose")
        self.assertGreaterEqual(len(chunks), 2)

    def test_crlf_normalized(self):
        src = "def alpha():\r\n    return 1\r\n"
        chunks = rt.chunk_text("m.py", src, ".py")
        self.assertIn("return 1", chunks[0]["text"])
        self.assertNotIn("\r", chunks[0]["text"])

    def test_long_function_split(self):
        body = "\n".join("    line_%d = %d" % (i, i) for i in range(200))
        src = "def big():\n" + body + "\n"
        chunks = rt.chunk_text("m.py", src, ".py")
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c["end"] - c["start"] + 1, rt.MAX_CHUNK_LINES)

    def test_comment_not_anchor(self):
        src = "# def fake():\ndef real():\n    return 1\n"
        chunks = rt.chunk_text("m.py", src, ".py")
        syms = [s for c in chunks for s in c["symbols"]]
        self.assertIn("real", syms)
        self.assertNotIn("fake", syms)


class Bm25(unittest.TestCase):
    def _index(self):
        chunks = [
            rt._chunk("a.py", 1, 3, "def invoice_total(items):\n    return sum(items)", "code", ["invoice_total"]),
            rt._chunk("b.py", 1, 2, "banana bread recipe flour sugar", "code", []),
            rt._chunk("c.py", 1, 2, "invoice invoice invoice invoice total total", "code", []),
        ]
        return rt.Index("/repo", chunks)

    def test_idf_rare_higher(self):
        idx = self._index()
        self.assertGreater(idx.idf("banana"), idx.idf("invoice"))

    def test_relevant_chunk_scores_first(self):
        idx = self._index()
        # the real search() path passes identifiers, so the defining chunk is boosted
        scores = idx.score(["invoice", "total"], identifiers=["invoice_total"])
        order = sorted(range(idx.n), key=lambda i: -scores[i])
        self.assertEqual(idx.chunks[order[0]]["relpath"], "a.py")

    def test_symbol_boost(self):
        idx = self._index()
        with_boost = idx.score(["invoice", "total"], identifiers=["invoice_total"])
        without = idx.score(["invoice", "total"])
        a = next(i for i, c in enumerate(idx.chunks) if c["relpath"] == "a.py")
        self.assertGreater(with_boost[a], without[a])

    def test_filename_boost(self):
        idx = self._index()
        boosted = idx.score(["recipe"], filenames=["b.py"])
        plain = idx.score(["recipe"])
        b = next(i for i, c in enumerate(idx.chunks) if c["relpath"] == "b.py")
        self.assertGreater(boosted[b], plain[b])

    def test_length_normalization(self):
        # c.py repeats terms many times; BM25 saturates so it must not dominate
        # a single relevant occurrence in a short chunk by an unbounded margin.
        idx = self._index()
        scores = idx.score(["invoice"])
        c = next(i for i, ch in enumerate(idx.chunks) if ch["relpath"] == "c.py")
        a = next(i for i, ch in enumerate(idx.chunks) if ch["relpath"] == "a.py")
        self.assertLess(scores[c], scores[a] * 6)

    def test_stats(self):
        idx = self._index()
        st = idx.stats()
        self.assertEqual(st["files"], 3)
        self.assertEqual(st["chunks"], 3)
        self.assertIn("invoice_total", idx.symbols)


class Cache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self._write("a.py", "def alpha():\n    return 1\n")
        self._write("b.py", "def beta():\n    return 2\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, text):
        with open(os.path.join(self.root, name), "w") as fh:
            fh.write(text)

    def test_cache_written(self):
        _, meta = rt.build_index(self.root)
        self.assertTrue(meta["cache_written"])
        self.assertTrue(os.path.isfile(meta["cache"]))
        self.assertEqual(meta["reused"], 0)

    def test_cache_reuse(self):
        rt.build_index(self.root)
        _, meta = rt.build_index(self.root)
        self.assertEqual(meta["reused"], 2)

    def test_cache_invalidation_on_edit(self):
        rt.build_index(self.root)
        # change content AND size so the mtime+size signature differs
        self._write("a.py", "def alpha_changed():\n    return 999\n")
        _, meta = rt.build_index(self.root)
        self.assertEqual(meta["reused"], 1)
        res = rt.search(self.root, "alpha_changed")
        self.assertTrue(res)
        self.assertEqual(res[0]["relpath"], "a.py")

    def test_cache_drops_deleted(self):
        rt.build_index(self.root)
        os.remove(os.path.join(self.root, "b.py"))
        idx, meta = rt.build_index(self.root)
        self.assertEqual(meta["files"], 1)
        self.assertEqual(meta["reused"], 1)

    def test_rebuild_ignores_cache(self):
        rt.build_index(self.root)
        _, meta = rt.build_index(self.root, rebuild=True)
        self.assertEqual(meta["reused"], 0)

    def test_corrupt_cache_recovered(self):
        _, meta = rt.build_index(self.root)
        with open(meta["cache"], "w") as fh:
            fh.write("{ not json")
        idx, meta2 = rt.build_index(self.root)
        self.assertEqual(meta2["reused"], 0)
        self.assertGreater(idx.n, 0)

    def test_version_mismatch_rebuilt(self):
        _, meta = rt.build_index(self.root)
        with open(meta["cache"]) as fh:
            blob = json.load(fh)
        blob["version"] = -1
        with open(meta["cache"], "w") as fh:
            json.dump(blob, fh)
        _, meta2 = rt.build_index(self.root)
        self.assertEqual(meta2["reused"], 0)

    def test_custom_cache_dir(self):
        cdir = os.path.join(self.root, "custom_cache")
        _, meta = rt.build_index(self.root, cache_dir=cdir)
        self.assertTrue(meta["cache"].startswith(cdir))


class Search(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        with open(os.path.join(self.root, "billing.py"), "w") as fh:
            fh.write("def invoice_total(items):\n    return sum(items)\n")
        with open(os.path.join(self.root, "recipe.md"), "w") as fh:
            fh.write("banana bread recipe with flour and sugar\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_ranking(self):
        res = rt.search(self.root, "invoice total")
        self.assertTrue(res)
        self.assertEqual(res[0]["relpath"], "billing.py")
        self.assertGreater(res[0]["score"], 0)

    def test_topk_limit(self):
        res = rt.search(self.root, "invoice total recipe banana", topk=1)
        self.assertEqual(len(res), 1)

    def test_empty_query(self):
        self.assertEqual(rt.search(self.root, "ab"), [])

    def test_no_match(self):
        self.assertEqual(rt.search(self.root, "zzzznotpresent"), [])

    def test_result_fields(self):
        res = rt.search(self.root, "invoice_total")
        r = res[0]
        for key in ("relpath", "start", "end", "score", "text", "symbols", "kind"):
            self.assertIn(key, r)

    def test_camel_query_matches_snake_doc(self):
        res = rt.search(self.root, "invoiceTotal")
        self.assertTrue(res)
        self.assertEqual(res[0]["relpath"], "billing.py")


class PackAndSelect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        with open(os.path.join(self.root, "a.py"), "w") as fh:
            fh.write("def invoice_total(items):\n    return sum(items)\n")
        with open(os.path.join(self.root, "b.md"), "w") as fh:
            fh.write("banana bread recipe with flour and sugar\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_pack_citations(self):
        res = rt.search(self.root, "invoice total")
        packed = rt.pack_context(res)
        self.assertIn("--- a.py:", packed)

    def test_pack_budget(self):
        res = rt.search(self.root, "invoice total recipe banana", topk=5)
        packed = rt.pack_context(res, max_chars=40)
        # at least the first block always fits; total stays near budget
        self.assertTrue(packed)
        self.assertLessEqual(len(packed), 40 + len(res[0]["text"]) + 40)

    def test_select_context_compat(self):
        ctx = rt.select_context(self.root, "invoice total", topk=2)
        self.assertTrue(ctx)
        self.assertEqual(ctx[0][0], "a.py")
        self.assertIn("a.py:", ctx[0][1])

    def test_select_context_empty(self):
        self.assertEqual(rt.select_context(self.root, "ab"), [])

    def test_select_context_max_chars(self):
        ctx = rt.select_context(self.root, "invoice total", topk=2, max_chars=20)
        for _, snip in ctx:
            self.assertLessEqual(len(snip), 20)


if __name__ == "__main__":
    unittest.main()