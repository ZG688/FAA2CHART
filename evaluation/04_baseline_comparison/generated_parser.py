# -*- coding: utf-8 -*-
# LLM-generated parser (codegen baseline), please review manually
import re, json

def parse_airway(code: str, content: str) -> dict:
    result = {
        "airway_code": code,
        "MEA": "none",
        "MAA": "none",
        "airway_point": []
    }
    if not content:
        return result
    if '<table>' not in content:
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^([^,]+),\s*([A-Z]{2})\s+(.+)$', line)
            if m:
                name = m.group(1).strip()
                region = m.group(2).strip()
                typ = m.group(3).strip()
                result["airway_point"].append({
                    "name": name,
                    "region": region,
                    "type": typ,
                    "position": ["none", "none"]
                })
        return result
    table_pattern = re.compile(r'<table>(.*?)</table>', re.DOTALL | re.IGNORECASE)
    table_match = table_pattern.search(content)
    if not table_match:
        return result
    table_content = table_match.group(1)
    tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
    trs = tr_pattern.findall(table_content)
    points = []
    mea_vals = []
    maa_vals = []
    for tr in trs:
        td_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL | re.IGNORECASE)
        tds = td_pattern.findall(tr)
        if not tds:
            continue
        first_td = tds[0].strip() if tds else ''
        if re.search(r'MOCA|MRA|GNSS MEA|HF COMMS|HF COMMUNICATIONS|VHF/UHF COMMS|VHF COMMS|\*FOR THAT AIRSPACE', first_td, re.IGNORECASE):
            continue
        if re.search(r'colspan', tr, re.IGNORECASE):
            continue
        if len(tds) >= 3:
            mea_text = tds[2].strip() if tds[2] else ''
            mea_match = re.search(r'\d{3,5}', mea_text)
            if mea_match:
                mea_vals.append(mea_match.group())
        if len(tds) >= 4:
            maa_text = tds[3].strip() if tds[3] else ''
            maa_match = re.search(r'\d{3,5}', maa_text)
            if maa_match:
                maa_vals.append(maa_match.group())
        for idx in [0, 1]:
            if idx >= len(tds):
                continue
            cell_text = tds[idx].strip()
            if not cell_text:
                continue
            lines = cell_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if re.search(r'MOCA|MRA|GNSS MEA|HF COMMS|HF COMMUNICATIONS|VHF/UHF COMMS|VHF COMMS|\*FOR THAT AIRSPACE', line, re.IGNORECASE):
                    continue
                if re.match(r'^\*?\d{3,5}', line):
                    continue
                m = re.match(r'^([^,]+),\s*([A-Z]{2})\s+(.+)$', line)
                if m:
                    name = m.group(1).strip()
                    region = m.group(2).strip()
                    typ = m.group(3).strip()
                    point = {
                        "name": name,
                        "region": region,
                        "type": typ,
                        "position": ["none", "none"]
                    }
                    if not points or points[-1] != point:
                        points.append(point)
    unique_points = []
    for p in points:
        if not unique_points or p != unique_points[-1]:
            unique_points.append(p)
    result["airway_point"] = unique_points
    if mea_vals:
        result["MEA"] = mea_vals[0]
    if maa_vals:
        result["MAA"] = maa_vals[0]
    return result