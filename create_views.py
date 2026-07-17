from step_views import create_step_views_for_folder
from dataset import DATA_STEP, DATA_VIEWS

result = create_step_views_for_folder(
    DATA_STEP,
    DATA_VIEWS,
    number_of_views=12,
    resolution=(1920, 1080),
    elevation=20,
    skip_existing=True,
)

print(f"Created {len(result.created_images):,} images")
print(f"Skipped {len(result.skipped_files):,} completed STEP files")
print(f"Failed {len(result.failed_files):,} STEP files")
