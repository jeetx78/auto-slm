import os
import uuid
import click

from autoslm.task.infer import infer_task_from_dataset
from autoslm.supabase_client import supabase
from autoslm.data.ingest import ingest_directory
from autoslm.training.config import training_config_for_task
from autoslm.training.lora_train import main as train_lora


@click.group()
def cli():
    """Auto-SLM: Automatic Small Language Model Factory"""
    pass


# ------------------------
# INIT COMMAND
# ------------------------
@cli.command()
@click.option("--name", required=True)
def init(name):
    """Create a new Auto-SLM project"""

    res = supabase.table("projects").insert({"name": name}).execute()

    if not res.data:
        raise click.ClickException("Project creation failed")

    project_id = res.data[0]["id"]
    click.echo("✅ Project created successfully")
    click.echo(f"📌 Project ID: {project_id}")


# ------------------------
# TRAIN (DATA ONLY)
# ------------------------
@cli.command()
@click.option("--project-id", required=True)
@click.option("--data", required=True)
@click.option("--model", default="phi-3-mini")
def train(project_id, data, model):
    """Ingest data and register training run"""

    if not os.path.exists(data):
        raise click.ClickException(f"Data path does not exist: {data}")

    click.echo("📥 Ingesting data...")
    result = ingest_directory(data, "./artifacts")

    task = infer_task_from_dataset(result["output"])
    click.echo(f"🧠 Inferred task type: {task}")

    supabase.table("datasets").insert({
        "project_id": project_id,
        "source": os.path.abspath(data),
        "num_files": result["files"]
    }).execute()

    run = supabase.table("training_runs").insert({
        "project_id": project_id,
        "model_name": model,
        "status": "data_ready",
        "task_type": task
    }).execute()

    config = training_config_for_task(task)
    click.echo("⚙️ Training configuration:")
    for k, v in config.items():
        click.echo(f"   {k}: {v}")

    click.echo(f"🚀 Training run registered: {run.data[0]['id']}")


# ------------------------
# BUILD (FULL PIPELINE)
# ------------------------
@cli.command()
@click.option("--project", required=True, help="Project name or UUID")
@click.option("--data", required=True, help="Path to data directory")
@click.option("--model", default="sshleifer/tiny-gpt2")
def build(project, data, model):
    """Full Auto-SLM pipeline"""

    # Resolve project safely (UUID-aware)
    try:
        uuid.UUID(project)
        is_uuid = True
    except ValueError:
        is_uuid = False

    if is_uuid:
        res = supabase.table("projects").select("*").eq("id", project).execute()
    else:
        res = supabase.table("projects").select("*").eq("name", project).execute()

    if not res.data:
        raise click.ClickException("Project not found")

    project_id = res.data[0]["id"]

    click.echo("🚀 Auto-SLM build started")

    # 1. Ingest
    click.echo("📥 Ingesting data...")
    result = ingest_directory(data, "./artifacts")

    # 2. Infer task
    task = infer_task_from_dataset(result["output"])
    click.echo(f"🧠 Task inferred: {task}")

    # 3. Training config
    config = training_config_for_task(task)
    click.echo("⚙️ Training config:")
    for k, v in config.items():
        click.echo(f"   {k}: {v}")

    # 4. Save run
    supabase.table("training_runs").insert({
        "project_id": project_id,
        "model_name": model,
        "task_type": task,
        "status": "training"
    }).execute()

    # 5. Train LoRA
    if os.path.exists("artifacts/adapter/adapter_model.safetensors"):
        click.echo("♻️ Adapter already exists, skipping training")
    else:
        click.echo("🔥 Training LoRA adapter...")
        train_lora()

    click.echo("✅ Auto-SLM build complete")
    click.echo("👉 Run: python -m uvicorn autoslm.serve:app")


if __name__ == "__main__":
    cli()
