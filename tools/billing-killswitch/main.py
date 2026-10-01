"""
billing-killswitch — hard spend cap for the festival-reviewer project.

Cloud Billing budgets only NOTIFY; they never stop spending. This function is
what turns the Rs 7,000 budget into an actual cap: the budget publishes every
threshold crossing to Pub/Sub, and when reported spend reaches the budget
amount this unlinks the billing account from the project, which halts all
billable services.

Read this before relying on it:

  * Unlinking billing is abrupt, not a throttle. Cloud Run stops serving (503)
    and any review being judged at that moment dies. This is the same state the
    project was in on 2026-09-30.
  * Budget data lags by hours, so the trip can land past Rs 7,000, not exactly on it.
  * Recovery is manual: relink the billing account in the console. Nothing here
    re-enables it.

Deployed as a gen2 Cloud Function triggered by the billing-killswitch topic.
"""
import base64
import json
import os

import googleapiclient.discovery

TARGET_PROJECT = os.environ.get("TARGET_PROJECT", "")
PROJECT_NAME = f"projects/{TARGET_PROJECT}"


def _billing():
    return googleapiclient.discovery.build("cloudbilling", "v1", cache_discovery=False)


def _payload(event) -> dict:
    """Pull the budget notification out of whatever shape we are handed.

    Pub/Sub-triggered functions arrive either as a CloudEvent object
    (event.data["message"]["data"]) or as a legacy background event
    (event["data"]). Both are accepted so a framework or deploy-flag change
    cannot quietly stop this from firing — which is the one failure mode a
    spend cap must not have.
    """
    raw = ""
    data = getattr(event, "data", None)
    if isinstance(data, dict):
        if isinstance(data.get("message"), dict):
            raw = data["message"].get("data", "")
        elif "data" in data:
            raw = data.get("data", "")
    elif isinstance(event, dict):
        if isinstance(event.get("message"), dict):
            raw = event["message"].get("data", "")
        else:
            raw = event.get("data", "")
    if not raw:
        return {}
    return json.loads(base64.b64decode(raw).decode("utf-8"))


def stop_billing(event, context=None):
    """Pub/Sub entry point. Disables billing once spend >= budget."""
    if not TARGET_PROJECT:
        print("TARGET_PROJECT not set — refusing to act")
        return

    notice = _payload(event)
    if not notice:
        print("[skip] no decodable budget payload")
        return
    cost = float(notice.get("costAmount", 0) or 0)
    budget = float(notice.get("budgetAmount", 0) or 0)
    currency = notice.get("currencyCode", "")

    # Every threshold crossing publishes here, including the warnings at
    # 4k/5k/6k. Only the final one should actually cut billing.
    if budget <= 0 or cost < budget:
        print(f"[ok] {cost} {currency} of {budget} — warning only, no action")
        return

    projects = _billing().projects()
    info = projects.getBillingInfo(name=PROJECT_NAME).execute()
    if not info.get("billingEnabled"):
        print(f"[noop] billing already disabled on {PROJECT_NAME}")
        return

    projects.updateBillingInfo(
        name=PROJECT_NAME, body={"billingAccountName": ""}
    ).execute()
    print(f"[STOPPED] billing disabled on {PROJECT_NAME} at {cost} {currency} "
          f"(cap {budget}). Relink the billing account manually to restore service.")
