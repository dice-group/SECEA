import io

import lightrdf
import torch

from decorators import timed
from unidecode import unidecode
rdf_type_uri = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
owl_class_uri = '<http://www.w3.org/2002/07/owl#Class>'
iri_label_uri = '<iri>'
wiki_page_link_uri = '<http://dbkwik.webdatacommons.org/ontology/wikiPageWikiLink>'

class Dataset:
    @timed
    def __init__(self, clean_text: str, suffix: str, add_iri_label : bool = True):
        self.add_iri_label = add_iri_label
        self.anything_to_id = dict()
        self.id_to_anything = dict()
        self.entity_to_id = None
        self.id_to_entity = None
        self.literal_to_id = None
        self.id_to_literal = None
        self.class_to_id = None
        self.id_to_class = None
        self.rel_to_id = None
        self.id_to_rel = None
        self._triples_raw_list = list()
        self._is_literal_list: list[bool] = []
        self.triples_tensor : None | torch.IntTensor = None
        self.entity_id_mask : None | torch.IntTensor = None
        self.literal_id_mask : None | torch.IntTensor = None
        self.class_id_mask : None | torch.IntTensor = None
        self.rel_id_mask : None | torch.IntTensor = None
        self.anything_to_type_id = None
        self.literal_cache_for_dataset_building = dict()

        print("Load dataset")
        byte_stream = io.BytesIO(clean_text.encode("utf-8"))
        if suffix in [".rdf", ".xml", ".owl", ".rdfxml"]:
            parser = lightrdf.xml.PatternParser
        elif suffix in [".ttl", ".turtle"]:
            parser = lightrdf.turtle.PatternParser
        elif suffix in [".nt", ".ntriples"]:
            parser = lightrdf.nt.PatternParser
        else:
            raise ValueError(f"Unsupported RDF format: {suffix}")

        doc = lightrdf.RDFDocument(byte_stream, parser=parser)
        for triple in doc.search_triples(None, None, None):
            self.add_triple(triple)
        print("Added all triples")
        self.on_done_loading()

    def _get_or_create_id(self, iri, is_literal : bool =False):
        idx = self.anything_to_id.get(iri)
        if idx is None:
            idx = len(self.anything_to_id)
            self.anything_to_id[iri] = idx
            self.id_to_anything[idx] = iri
            self._is_literal_list.append(bool(is_literal))
        return idx

    def add_triple(self, triple):
        head, rel, tail = triple

        tail_is_literal = False
        if tail.startswith('"'):
            tail_is_literal = True
            if tail in self.literal_cache_for_dataset_building:
                tail = self.literal_cache_for_dataset_building[tail]
            else:
                processed_tail = unidecode(tail[1:-1].split("^^", 1)[0])
                self.literal_cache_for_dataset_building[tail] = processed_tail
                tail = processed_tail

        h = self._get_or_create_id(head)
        r = self._get_or_create_id(rel)
        t = self._get_or_create_id(tail, tail_is_literal)

        self._triples_raw_list.append((h, r, t))

    @timed
    def add_iri_labels(self):
        new_triples = []

        for iri, idx in list(self.anything_to_id.items()):
            if self._is_literal_list[idx]:
                continue
            if iri == iri_label_uri:
                continue

            last_part = iri[max(iri.rfind('/'), iri.rfind('#')) + 1:].strip('<>')

            h = idx
            r = self._get_or_create_id(iri_label_uri)
            t = self._get_or_create_id(last_part, is_literal=True)

            new_triples.append((h, r, t))
        self._triples_raw_list.extend(new_triples)

    @timed
    def on_done_loading(self):
        if self.add_iri_label:
            self.add_iri_labels()
        del self.literal_cache_for_dataset_building
        self.triples_tensor = torch.tensor(self._triples_raw_list, dtype=torch.long)
        del self._triples_raw_list
        print("Created triples tensor")
        self.identify_entities_and_literals_and_classes_and_properties()

        entity_ids = torch.nonzero(self.entity_id_mask, as_tuple=True)[0]
        literal_ids = torch.nonzero(self.literal_id_mask, as_tuple=True)[0]
        class_ids = torch.nonzero(self.class_id_mask, as_tuple=True)[0]
        rel_ids = torch.nonzero(self.rel_id_mask, as_tuple=True)[0]

        self.id_to_entity = {i: self.id_to_anything[i] for i in entity_ids.tolist()}
        self.entity_to_id = {v: k for k, v in self.id_to_entity.items()}

        self.id_to_literal = {i: self.id_to_anything[i] for i in literal_ids.tolist()}
        self.literal_to_id = {v: k for k, v in self.id_to_literal.items()}

        self.id_to_class = {i: self.id_to_anything[i] for i in class_ids.tolist()}
        self.class_to_id = {v: k for k, v in self.id_to_class.items()}

        self.id_to_rel = {i: self.id_to_anything[i] for i in rel_ids.tolist()}
        self.rel_to_id = {v: k for k, v in self.id_to_rel.items()}

        #Must be after rel_to_id
        self.create_anything_to_type_mapping()

    @timed
    def create_anything_to_type_mapping(self):
        rdf_type_id = self.rel_to_id.get(rdf_type_uri)
        if rdf_type_id is not None:
            self.anything_to_type_id = -torch.ones(len(self.anything_to_id), dtype=torch.long)
            type_triples = self.triples_tensor[self.triples_tensor[:, 1] == rdf_type_id]
            self.anything_to_type_id[type_triples[:, 0]] = type_triples[:, 2]

    @timed
    def identify_entities_and_literals_and_classes_and_properties(self):
        self.literal_id_mask = torch.tensor(self._is_literal_list, dtype=torch.bool)
        del self._is_literal_list
        print("Created literal mask.")
        num_ids = len(self.anything_to_id)
        self.rel_id_mask = torch.zeros(num_ids, dtype=torch.bool)
        rel_indices = self.triples_tensor[:, 1].contiguous()
        self.rel_id_mask[rel_indices] = True
        print("Created rel mask.")

        self.class_id_mask = torch.zeros(num_ids, dtype=torch.bool) #Classes are things that are referred to at least once by an rdf:type triple or are a rdf:type '<http://www.w3.org/2002/07/owl#Class>'
        rdf_type_id = self.anything_to_id.get(rdf_type_uri)
        if rdf_type_id is not None:
            type_triple_mask = self.triples_tensor[:, 1] == rdf_type_id
            classes = self.triples_tensor[type_triple_mask, 2]
            self.class_id_mask[classes] = True

            owl_class_id = self.anything_to_id.get(owl_class_uri)
            if owl_class_id is not None:
                explicit_classes = self.triples_tensor[type_triple_mask & (self.triples_tensor[:, 2] == owl_class_id), 0]
                self.class_id_mask[explicit_classes] = True
        print("Created class mask.")
        self.entity_id_mask = ~(self.class_id_mask | self.rel_id_mask | self.literal_id_mask)
        print("Created entity mask.")