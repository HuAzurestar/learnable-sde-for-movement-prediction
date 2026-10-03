"""Bounded case figure kernel; production rendering runs only in a managed worker."""

import hashlib
import math
from xml.etree import ElementTree as ET

from application.research_figures import NS, MAX_FIGURE_BYTES, validate_figure_svg
from infrastructure.research_store import ResearchError, digest, encode

MAX_OPERATIONS = 20_000_000
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024


def case_plan(result, max_operations=MAX_OPERATIONS):
    try:
        if type(max_operations) is not int or not 0 < max_operations <= MAX_OPERATIONS:
            raise ResearchError('RESOURCE_PLAN_REJECTED', 'invalid case graph operation quota')
        if not isinstance(result, dict) or len(encode(result)) > MAX_SOURCE_BYTES:
            raise ResearchError('RESOURCE_PLAN_REJECTED', 'case input exceeds byte quota')
        forecast = result.get('forecast', {})
        if not isinstance(forecast, dict):
            raise ValueError('forecast must be an object')
        samples = forecast.get('samples')
        if samples is None or samples == []:
            return None  # Never fabricate paths from per-segment scores.
        if not isinstance(samples, list) or not 1 <= len(samples) <= 64:
            raise ResearchError('RESOURCE_PLAN_REJECTED', 'case preview exceeds 64 saved paths')
        order, units, horizons = result['state_order'], result['units'], forecast['horizons']
        if order not in (['x', 'vx'], ['x', 'y'], ['x', 'y', 'vx', 'vy']):
            raise ValueError('unsupported saved state order')
        if (not isinstance(units, list) or len(units) != len(order) or
                any(not isinstance(unit, str) or not 0 < len(unit) <= 64 for unit in units) or
                not isinstance(result.get('time_unit'), str) or not 0 < len(result['time_unit']) <= 64):
            raise ValueError('saved state/time units missing')
        if not isinstance(horizons, list) or not 1 <= len(horizons) <= 512:
            raise ResearchError('RESOURCE_PLAN_REJECTED', 'case horizon quota exceeded')
        if (any(type(h) not in (int, float) or not math.isfinite(h) or h <= 0 for h in horizons) or
                any(a >= b for a, b in zip(horizons, horizons[1:]))):
            raise ValueError('saved horizons must increase strictly')
        for path in samples:
            if not isinstance(path, list) or len(path) != len(horizons):
                raise ValueError('saved path and horizon lengths differ')
            for point in path:
                if (not isinstance(point, list) or len(point) != len(order) or
                        any(type(v) not in (int, float) or not math.isfinite(v) for v in point)):
                    raise ValueError('saved point state width or finite values differ')
        policy = forecast['preview']
        if (not isinstance(policy, dict) or len(encode(policy)) > 16384 or
                any(not isinstance(policy.get(key), str) or not 0 < len(policy[key]) <= 256
                    for key in ('case_selection_rule', 'generation_version')) or
                type(policy.get('n_samples')) is not int or policy['n_samples'] != len(samples) or
                not isinstance(policy.get('sample_ids'), list) or len(policy['sample_ids']) != len(samples) or
                any(type(v) not in (str, int) or len(encode(v)) > 256 for v in policy['sample_ids']) or
                len({encode(v) for v in policy['sample_ids']}) != len(samples)):
            raise ValueError('frozen preview selection policy missing or inconsistent')
        panels = len(order) // 2
        points = panels * len(samples) * (len(horizons) + len(horizons) * (len(horizons) + 1) // 2)
        figures = len(horizons) + 1
        operations = points * 8 + len(samples) * len(horizons) * len(order) + figures * 256
        # Provenance is embedded in XML text and that SVG is embedded in JSON.
        # Budget the actual frozen policy, including XML entity expansion and
        # a second conservative JSON escaping allowance, before any rendering.
        policy_bytes = encode(policy)
        escaped_policy_bytes = (len(policy_bytes) + 4 * policy_bytes.count(b'&') +
                                3 * policy_bytes.count(b'<') + 3 * policy_bytes.count(b'>'))
        output_bound = points * 128 + figures * (2 * escaped_policy_bytes + 16384)
        if operations > max_operations or output_bound > MAX_OUTPUT_BYTES:
            raise ResearchError('RESOURCE_PLAN_REJECTED', 'joint case graph allocation exceeds quota')
        return {'schema_version': 'pirc25-case-graph-plan-v1', 'renderer_version': 'saved-paths-v1',
                'paths': len(samples), 'horizons': len(horizons), 'panels': panels,
                'figure_count': figures, 'maximum_operations': max_operations,
                'planned_operations': operations, 'maximum_input_bytes': MAX_SOURCE_BYTES,
                'maximum_output_bytes': MAX_OUTPUT_BYTES, 'figure_byte_bound': MAX_FIGURE_BYTES,
                'planned_output_bytes': output_bound, 'allocation': 'shared-job-on-source-arm'}
    except ResearchError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ResearchError('CONTRACT_MISMATCH', 'saved case layout or preview policy differs') from exc


def provenance(result, source_id, reference, selection):
    return {'schema_version': 'pirc25-figure-provenance-v1', 'kind': 'case',
            'source_artifact_id': source_id, 'spec_hash': result['spec_hash'], 'cell_hash': result['cell_hash'],
            'protocol_hash': result['protocol_hash'], 'state_order': result['state_order'], 'units': result['units'],
            'time_unit': result['time_unit'], 'horizon_index': selection,
            'horizon': None if selection is None else result['forecast']['horizons'][selection],
            'preview_policy': result['forecast']['preview'], 'computation_ref': reference,
            'cost_source': 'resolve-computation-ref-after-settlement'}


def render_case_package(result, source_id, reference, max_operations=MAX_OPERATIONS):
    plan = case_plan(result, max_operations)
    if plan is None:
        raise ResearchError('MISSING_INPUT', 'no saved sample paths for case graphics')
    ET.register_namespace('', NS)
    samples = result['forecast']['samples']
    bounds = []
    for axis in range(len(result['state_order'])):
        values = [point[axis] for path in samples for point in path]
        low, high = min(values), max(values)
        scale = max(abs(low), abs(high), 1)
        bounds.append((low / scale, high / scale, scale))

    def coordinate(value, axis):
        low, high, scale = bounds[axis]
        return (value / scale - low) / (high - low or 1)

    figures, entries = {}, []
    for selection in [None, *range(plan['horizons'])]:
        root = ET.Element('{' + NS + '}svg', {'viewBox': f"0 0 600 {300 * plan['panels'] + 60}",
                          'role': 'img', 'aria-label': 'Frozen saved forecast paths; display only'})
        ET.SubElement(root, '{' + NS + '}metadata').text = encode(provenance(result, source_id, reference, selection)).decode()
        for panel in range(plan['panels']):
            a, b = panel * 2, panel * 2 + 1
            for path in samples:
                visible = path if selection is None else path[:selection + 1]
                points = ' '.join(f"{40 + coordinate(p[a], a) * 520:.6g},{panel * 300 + 250 - coordinate(p[b], b) * 220:.6g}" for p in visible)
                ET.SubElement(root, '{' + NS + '}polyline', {'points': points, 'fill': 'none',
                    'stroke': '#287c9c', 'stroke-opacity': '.4'})
            ET.SubElement(root, '{' + NS + '}text', {'x': '40', 'y': str(panel * 300 + 285)}).text = (
                f"{result['state_order'][a]} ({result['units'][a]}) / {result['state_order'][b]} ({result['units'][b]}) · {len(samples)} saved paths · display only")
        ET.SubElement(root, '{' + NS + '}text', {'x': '12', 'y': str(plan['panels'] * 300 + 20), 'font-size': '9'}).text = 'source_artifact_id: ' + source_id
        content = ET.tostring(root, encoding='utf-8')
        if len(content) > MAX_FIGURE_BYTES:
            raise ResearchError('RESOURCE_PLAN_REJECTED', 'case figure byte quota exceeded')
        sha = hashlib.sha256(content).hexdigest()
        name = 'case-figure-' + sha + '.svg'
        figures[name] = content.decode()
        entries.append({'filename': name, 'sha256': sha, 'size_bytes': len(content), 'media_type': 'image/svg+xml',
                        'kind': 'case', 'horizon_index': selection,
                        'horizon': None if selection is None else result['forecast']['horizons'][selection]})
    index = {'schema_version': 'pirc25-case-figure-index-v1', 'source_artifact_id': source_id,
             'spec_hash': result['spec_hash'], 'cell_hash': result['cell_hash'], 'protocol_hash': result['protocol_hash'],
             'computation_ref': reference, 'resource_plan': plan, 'figures': entries}
    if len(encode({'figure_index': index, 'figures': figures})) > MAX_OUTPUT_BYTES:
        raise ResearchError('RESOURCE_PLAN_REJECTED', 'case graph package byte quota exceeded')
    return {'figure_index': index, 'figures': figures}


def validate_case_package(result, source_id, index, figures):
    try:
        plan = case_plan(result, index['resource_plan']['maximum_operations'])
        expected = {'schema_version': 'pirc25-case-figure-index-v1', 'source_artifact_id': source_id,
                    'spec_hash': result['spec_hash'], 'cell_hash': result['cell_hash'],
                    'protocol_hash': result['protocol_hash'], 'computation_ref': index['computation_ref'],
                    'resource_plan': plan, 'figures': index['figures']}
        if index != expected or plan is None or not isinstance(figures, dict):
            raise ValueError('case index binding differs')
        entries = index['figures']
        if not isinstance(entries, list) or len(entries) != plan['figure_count']:
            raise ValueError('case figure count differs')
        selections, names = set(), set()
        for entry in entries:
            selection = entry['horizon_index']
            if selection is not None and (type(selection) is not int or not 0 <= selection < plan['horizons']):
                raise ValueError('case horizon selector differs')
            content = figures[entry['filename']].encode('utf-8')
            sha = hashlib.sha256(content).hexdigest()
            if (entry != {'filename': 'case-figure-' + sha + '.svg', 'sha256': sha, 'size_bytes': len(content),
                          'media_type': 'image/svg+xml', 'kind': 'case', 'horizon_index': selection,
                          'horizon': None if selection is None else result['forecast']['horizons'][selection]} or
                    selection in selections or entry['filename'] in names or
                    validate_figure_svg(content, expected_kind='case') != provenance(result, source_id, index['computation_ref'], selection)):
                raise ValueError('case figure bytes/provenance differ')
            selections.add(selection)
            names.add(entry['filename'])
        if set(figures) != names or selections != {None, *range(plan['horizons'])}:
            raise ValueError('case package has missing or extra figures')
        if len(encode({'figure_index': index, 'figures': figures})) > MAX_OUTPUT_BYTES:
            raise ValueError('case package bytes exceeded')
        return digest(index)
    except ResearchError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ResearchError('CONTRACT_MISMATCH', 'frozen case figure package differs') from exc
