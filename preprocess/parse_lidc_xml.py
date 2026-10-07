"""Parse LIDC XML annotation files -> annotations keyed by SeriesInstanceUID.

Each XML file corresponds to ONE CT series and contains up to 4 <readingSession>
elements (one per radiologist). Folder names in the XML-only zip are internal
numbers, so we key everything by the <SeriesInstanceUid> tag read from inside
each XML, and match to DICOM series by that UID later.

Output (annotations.json):
  { "<SeriesInstanceUID>": { "sessions": [ {"doctor": ..., "nodules": [
      {"rois": [{"sop_uid": ..., "points": [[x,y],...]}, ...]}, ...]}, ... ] } }
"""
import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET

NS = "http://www.nih.gov"


def parse_xml(path):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    suid = root.find(f".//{{{NS}}}SeriesInstanceUid")
    if suid is None or not suid.text:
        return None
    sessions = []
    for session in root.iter(f"{{{NS}}}readingSession"):
        doctor = session.findtext(f"{{{NS}}}servicingRadiologistID") or "unknown"
        nodules = []
        for nod in session.iter(f"{{{NS}}}unblindedReadNodule"):
            rois = []
            for roi in nod.iter(f"{{{NS}}}roi"):
                inclusion = (roi.findtext(f"{{{NS}}}inclusion") or "TRUE").strip().upper() == "TRUE"
                if not inclusion:
                    continue
                sop = roi.findtext(f"{{{NS}}}imageSOP_UID")
                points = []
                for edge in roi.iter(f"{{{NS}}}edgeMap"):
                    x = edge.findtext(f"{{{NS}}}xCoord")
                    y = edge.findtext(f"{{{NS}}}yCoord")
                    if x is not None and y is not None:
                        points.append([float(x), float(y)])
                if sop and points:
                    rois.append({"sop_uid": sop, "points": points})
            if rois:
                nodules.append({"rois": rois})
        sessions.append({"doctor": doctor, "nodules": nodules})
    return suid.text, sessions


def main(raw_dir, out_path):
    out = {}
    for xml_path in glob.glob(os.path.join(raw_dir, "**", "*.xml"), recursive=True):
        res = parse_xml(xml_path)
        if res is None:
            continue
        suid, sessions = res
        is_resub = "resubmitted" in os.path.basename(xml_path).lower()
        existing = out.get(suid)
        if existing is None or (is_resub and "resubmitted" not in existing["_src"].lower()):
            out[suid] = {"sessions": sessions, "_src": os.path.basename(xml_path)}
    # drop internal _src
    out = {k: {"sessions": v["sessions"]} for k, v in out.items()}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"Parsed {len(out)} series -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out", default="data/processed/annotations.json")
    args = ap.parse_args()
    main(args.raw_dir, args.out)
