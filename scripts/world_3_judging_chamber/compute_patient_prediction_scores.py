def examine_neighbors():
    config = get_global_config()

    neighbors_by_patient = load_neighbors_for_config(config)
    predictions = load_all_patient_results(config)
    idf_registry = load_idf_registry()
    all_patients = {p["patient_id"]: p for p in load_patient_data()}

    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 🛠 Use config to name log and output files
    config_suffix = f"{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}_{config.num_neighbors}"
    log_path = os.path.join("logs", f"neighbor_analysis_errors_{config_suffix}.txt")
    output_path = os.path.join("data", f"neighbor_analysis_summary_{config_suffix}.json")

    log_lines = []
    output_data = []

    print("--- Examining neighbors ---")
    for patient_id, neighbors in neighbors_by_patient.items():
        patient = all_patients.get(patient_id)
        predicted = predictions.get(patient_id)

        if not patient or not predicted:
            log_lines.append(f"[SKIP] Missing data for patient {patient_id}")
            continue

        actual = patient["visits"][config.num_visits - 1]

        try:
            scores = score_prediction(predicted, actual, idf_registry)
            output_data.append({
                "patient_id": patient_id,
                "top_neighbors": [n[0][0] for n in neighbors[:3]],
                "prediction_scores": scores
            })
        except Exception as e:
            log_lines.append(f"[ERROR] Failed scoring for patient {patient_id}: {str(e)}")

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    print(f"✅ Done. {len(output_data)} patients processed.")
    print(f"📄 Output: {output_path}")
    print(f"🪵 Log: {log_path}")
