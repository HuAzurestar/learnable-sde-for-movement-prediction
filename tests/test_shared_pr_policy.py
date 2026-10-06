from scripts.check_pr_title import main


def test_existing_pr_has_narrow_issue_binding():
    assert main("#23 feat(runtime): Add shared evidence", "feature/pirc-38-shared-engineering") == 0
    assert main("#24 feat: Wrong issue", "feature/pirc-38-shared-engineering") == 1
    assert main("#23 fix: Wrong type", "feature/pirc-38-shared-engineering") == 1
    assert main("PIRC-38: draft", "feature/pirc-38-shared-engineering") == 1
    assert main("#24 feat: Normal feature", "feature/24-normal") == 0
    assert main("#24 feat: Unbound branch", "feature/pirc-39") == 1
