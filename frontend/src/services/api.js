import axios from "axios";

const api = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8001",
    timeout: 30000,
});

function friendly(err, fallback) {
    if (err.response?.data?.detail) return new Error(err.response.data.detail, { cause: err });
    if (err.code === "ECONNABORTED") return new Error("The server took too long to respond. Try again.", { cause: err });
    return new Error(fallback, { cause: err });
}

/**
 * Model card: classes, LOSO scores, per-class recall, confusion matrix.
 * Everything is read from the trained artifact, never hardcoded.
 */
export async function getModelCard() {
    try {
        const { data } = await api.get("/model");
        return data;
    } catch (err) {
        throw friendly(err, "Could not reach the backend. Is it running on port 8001?");
    }
}

/** A real dataset photo, so the page can be tried without an upload. */
export async function getSampleBlob(id) {
    try {
        const { data } = await api.get(`/samples/${id}`, { responseType: "blob" });
        return data;
    } catch (err) {
        throw friendly(err, "Could not load that sample photo.");
    }
}

/** POST multipart image to /predict. */
export async function predictImage(blob) {
    const form = new FormData();
    form.append("file", blob, "turmeric.jpg");
    try {
        const { data } = await api.post("/predict", form, {
            headers: { "Content-Type": "multipart/form-data" },
        });
        return data;
    } catch (err) {
        throw friendly(err, "Could not reach the classifier. Is the backend running?");
    }
}
