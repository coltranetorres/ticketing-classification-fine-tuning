"""Prompt template and output parser shared by all models."""
import re

LABEL_DEFINITIONS: dict[str, str] = {
    "cancel_order": "The customer wants to cancel an order they placed.",
    "change_order": "The customer wants to modify the contents or details of an existing order.",
    "change_shipping_address": "The customer wants to change the shipping address on an existing order.",
    "check_cancellation_fee": "The customer asks whether there is a fee or charge for cancelling, without asking to cancel now.",
    "check_invoice": "The customer wants to view or check the details of an invoice or bill.",
    "check_payment_methods": "The customer asks which payment methods are accepted.",
    "check_refund_policy": "The customer asks about the company's refund rules or policy.",
    "complaint": "The customer wants to file a complaint or express dissatisfaction.",
    "contact_customer_service": "The customer asks how to reach customer service, such as its phone number, hours or email.",
    "contact_human_agent": "The customer asks to speak with a live human agent instead of an automated assistant.",
    "create_account": "The customer wants to create or open a new account.",
    "delete_account": "The customer wants to delete or close their account.",
    "delivery_options": "The customer asks what delivery or shipping options are available.",
    "delivery_period": "The customer asks how long delivery will take or when it will arrive.",
    "edit_account": "The customer wants to update or edit information on their existing account.",
    "get_invoice": "The customer asks to be sent or to download a copy of an invoice.",
    "get_refund": "The customer asks to receive a refund for a purchase.",
    "newsletter_subscription": "The customer wants to subscribe to or unsubscribe from the newsletter.",
    "payment_issue": "The customer reports a problem making a payment, such as a failed or declined payment.",
    "place_order": "The customer wants to buy something or place a new order.",
    "recover_password": "The customer forgot or lost their password and wants to recover or reset it.",
    "registration_problems": "The customer has trouble signing up or registering for an account.",
    "review": "The customer wants to leave or write a review or feedback about a product or service.",
    "set_up_shipping_address": "The customer wants to add or set up a shipping address for the first time.",
    "switch_account": "The customer wants to switch between multiple accounts or change which account they use.",
    "track_order": "The customer wants to know the current status or location of an order.",
    "track_refund": "The customer asks about the status of a refund they have already requested.",
}


def build_prompt(
    utterance: str,
    labels: list[str],
    definitions: dict[str, str] | None = None,
) -> str:
    lines = [
        "Classify the customer message into exactly one of the following intent labels.",
        "",
        "Labels:",
    ]
    for label in labels:
        if definitions:
            lines.append(f"{label}: {definitions[label]}")
        else:
            lines.append(label)
    lines += ["", f"Customer message: {utterance}", "", "Answer with exactly one label."]
    return "\n".join(lines)


def parse_label(text: str | None, labels: list[str]) -> str | None:
    if not text:
        return None
    low = text.lower()
    found = {
        label
        for label in labels
        if re.search(rf"(?<![a-z_]){re.escape(label.lower())}(?![a-z_])", low)
    }
    return found.pop() if len(found) == 1 else None
