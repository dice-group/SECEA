import traceback
from pathlib import Path

from flask import Flask, request, Response

from secea.alignment import Alignment
from secea.dataset import Dataset, iri_label_uri
from secea.secea import SECEA
from utils import timed, load_datasets_from_bytes
from xml.sax.saxutils import escape

app = Flask(__name__)

#This part allows our approach to run like a HTTP MELT Matcher
@app.route("/match", methods=["POST"])
def match():
    try:
        content_type = request.content_type or ""
        if "multipart/form-data" in content_type:

            if "source" not in request.files or "target" not in request.files:
                return "No source or target provided", 400

            source_file = request.files["source"]
            target_file = request.files["target"]

            source_dataset, target_dataset = load_datasets_from_bytes(source_file.read(), target_file.read(),
                                                  Path(source_file.filename).suffix.lower(),
                                                  Path(target_file.filename).suffix.lower(), multithread=False, add_iri_label=True)

            secea = SECEA(source_dataset, target_dataset)
            alignment = secea.get_alignment()

            alignment_xml = create_alignment_xml(alignment, source_dataset, target_dataset)

            # Return XML
            return Response(alignment_xml, mimetype="application/xml")
        else:
            return "Unsupported Content-Type", 400

    except Exception as e:
        return str(e), 500

@timed
def create_alignment_xml(alignment: Alignment, source: Dataset, target: Dataset):
    #Create an output XML like MELT matchers usually do
    xml_parts = []
    xml_parts.append("""<?xml version="1.0" encoding="utf-8"?>
<rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment#"
  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  xmlns:xsd="http://www.w3.org/2001/XMLSchema#">
<Alignment>
  <xml>yes</xml>
  <level>0</level>
  <type>11</type>
""")

    mappings = alignment.source_to_target_mappings

    for source_id in range(len(mappings)):
        target_id = mappings[source_id].item()

        if target_id == -1:
            #Unmatched things
            continue

        #Do not report literal matches
        if source_id in source.id_to_literal:
            continue
        if target_id in target.id_to_literal:
            continue

        #Transform IRIs back to MELT format
        raw_source_iri = source.id_to_anything[source_id]
        raw_target_iri = target.id_to_anything[target_id]
        if raw_source_iri == iri_label_uri:
            continue
        if raw_source_iri.startswith("<") and raw_source_iri.endswith(">"):
            raw_source_iri = raw_source_iri[1:-1]
        if raw_target_iri.startswith("<") and raw_target_iri.endswith(">"):
            raw_target_iri = raw_target_iri[1:-1]
        source_iri = escape(raw_source_iri, {'"': "&quot;"})
        target_iri = escape(raw_target_iri, {'"': "&quot;"})
        xml_parts.append(f"""  <map>
    <Cell>
      <entity1 rdf:resource="{source_iri}"/>
      <entity2 rdf:resource="{target_iri}"/>
      <relation>=</relation>
      <measure rdf:datatype="xsd:float">1.0</measure>
    </Cell>
  </map>
""")
    xml_parts.append("</Alignment>\n</rdf:RDF>\n")
    return "".join(xml_parts)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)