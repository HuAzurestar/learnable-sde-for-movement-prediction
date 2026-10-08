"""Consume completed checkpoint results into LOCAL review artifacts.

Never launches/stops predictions, refits, retries experiments or approves
claims. Optional waiting reads only two progress indexes; original context
and scientific consumers are loaded only after their dependencies finish.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .checkpoint_resume import load
from .protocol_core import read_json, unpack

FORECAST_KINDS = frozenset(('scientific_forecast', 'same_grid_reference',
    'inertial_path', 'forecast_replay', 'runtime_cold', 'runtime_warmup', 'runtime_warm'))


def readiness(directory, score_progress, settings, imported):
    """Dependency observation, NOT result qualification or producer ownership."""
    progress = read_json(Path(directory)/'progress.json')
    if progress['settings_id'] != settings['settings_id']:
        raise ValueError('checkpoint settings changed while observing')
    expected = {w['work_id'] for w in settings['workloads'] if w['kind'] in FORECAST_KINDS}
    terminal = set(imported) | set(progress['completed']) | set(progress['failures'])
    scores = read_json(score_progress) if Path(score_progress).is_file() else dict(rows={}, blocks={})
    score_ids = {w['work_id'] for w in settings['workloads'] if w['kind']=='common_scores'}
    return dict(event='review_dependencies', unfinished_forecasts=len(expected-terminal),
        terminal_failure_records=len(progress['failures']), pending=progress['pending'] is not None,
        score_rows=len(scores['rows']), expected_score_rows=11368,
        complete_blocks=len(scores['blocks']), expected_blocks=58,
        ready=(not expected-terminal and progress['pending'] is None
               and len(scores['rows'])==11368 and set(scores['blocks'])==score_ids and len(score_ids)==58),
        new_forecasts=0, new_fits=0, human_accepted=False)


def wait_ready(directory, score_progress, *, follow=False, poll_seconds=60, report_seconds=600):
    if poll_seconds<=0 or report_seconds<=0:
        raise ValueError('positive poll/report intervals required')
    settings, imported = load(directory)  # Metadata only, ONCE; no arrays/models/maps.
    next_report = 0
    while True:
        state = readiness(directory, score_progress, settings, imported)
        now = time.monotonic()
        if state['ready'] or now>=next_report:
            print(json.dumps(state), flush=True)
            next_report = now+report_seconds
        if state['ready']:
            return state
        if not follow:
            raise RuntimeError('review dependencies unfinished; prediction/scoring processes untouched')
        time.sleep(poll_seconds)  # No lock/ownership wait and no scientific work.


def consume(directory, score_progress, catalog_path, catalog_sha256, output_directory,
            tsde_directory, *, software_fixture=False):
    """Original producers/consumers, not another experiment or claim gate."""
    # Check the cheap public input BEFORE potentially expensive final work.
    unpack(read_json(catalog_path), expected_sha256=catalog_sha256)
    tsde = Path(tsde_directory).resolve()
    for name in ('aggregate_pirc17.py','plot_pirc17.py'):
        if not (tsde/'scripts'/name).is_file():
            raise ValueError('declared TSDE public artifact consumers required')
    from .checkpoint_audit import analyze, audit
    from .checkpoint_export import export
    from .evidence_cards import generate
    print(json.dumps(dict(event='review_stage', stage='analysis')), flush=True)
    analysis = analyze(directory)  # Existing immutable analysis is reused.
    audit_pointer = Path(score_progress).parent/'audit.json'
    print(json.dumps(dict(event='review_stage', stage='audit', reuse_existing=audit_pointer.is_file())), flush=True)
    if audit_pointer.is_file():
        audit_binding = read_json(audit_pointer)
    else:
        audit_binding = audit(directory)
    # Export validates any reused audit against the actual current sources.
    # A corrupt/stale audit fails here; never silently repeat expensive work.
    print(json.dumps(dict(event='review_stage', stage='public_export')), flush=True)
    public = export(directory)
    print(json.dumps(dict(event='review_stage', stage='review_cards')), flush=True)
    output = Path(output_directory).resolve()
    cards = generate(public['path'], public['content_sha256'], catalog_path,
                     catalog_sha256, output/'cards')
    artifacts = {}
    for label, module in (('tables','aggregate_pirc17'),('figures','plot_pirc17')):
        print(json.dumps(dict(event='review_stage', stage=label)), flush=True)
        command = [sys.executable,'-m','scripts.'+module,'--cards',cards['path'],
                   '--sha256',cards['content_sha256'],'--output-directory',str(output/label)]
        if label=='figures' and software_fixture:
            command.append('--software-fixture')
        # cwd is explicit; remove caller's PYTHONPATH so the selected TSDE
        # consumer is imported, not another repository's scripts package.
        import os
        environment = dict(os.environ)
        environment.pop('PYTHONPATH',None)
        result = subprocess.run(command,cwd=tsde,env=environment,check=True,
            capture_output=True,text=True,encoding='utf-8')
        if result.stderr:
            print(result.stderr,file=sys.stderr,end='',flush=True)
        artifacts[label] = json.loads(result.stdout)
    return dict(event='review_artifacts_ready', analysis=analysis, audit=audit_binding,
        public_export=public, cards=cards, artifacts=artifacts,
        software_fixture_only=software_fixture, new_forecasts=0,new_fits=0,
        scientific_claim_authorized=False,test02_qualification_asserted=False,
        manuscript_written=False,human_accepted=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('directory','score-progress','catalog','output-directory','tsde-directory'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--catalog-sha256',required=True)
    parser.add_argument('--follow',action='store_true')
    parser.add_argument('--poll-seconds',type=float,default=60)
    parser.add_argument('--report-seconds',type=float,default=600)
    parser.add_argument('--software-fixture',action='store_true')
    args = parser.parse_args(argv)
    wait_ready(args.directory,args.score_progress,follow=args.follow,
        poll_seconds=args.poll_seconds,report_seconds=args.report_seconds)
    print(json.dumps(consume(args.directory,args.score_progress,args.catalog,
        args.catalog_sha256,args.output_directory,args.tsde_directory,
        software_fixture=args.software_fixture)),flush=True)


if __name__=='__main__': main()
