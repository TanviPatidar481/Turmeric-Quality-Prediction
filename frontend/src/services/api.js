import axios from "axios";

const api = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000",
    timeout: 30000,
});

export const PRETTY_LABELS = {
    home_ground: "Home-ground (dried roots)",
    market_bought: "Market-bought",
};

export const RETAKE_TIPS =
    "Use bright diffuse light, fill frame with powder, no shadows/containers, hold steady.";

/**
 * POST multipart image to backend /predict.
 * Resolves to { label, confidence, probs?, warnings? }.
 * Throws Error with user-facing message on failure.
 */
export async function predictImage(blob) {
    const form = new FormData();
    form.append("file", blob, "turmeric.jpg");
    try {
        const { data } = await api.post("/predict", form, {
            headers: { "Content-Type": "multipart/form-data" },
        });
        return data;
    } catch (err) {
        if (err.response?.data?.detail) throw new Error(err.response.data.detail, { cause: err });
        if (err.code === "ECONNABORTED") throw new Error("Request timed out — try again.", { cause: err });
        throw new Error(
            "Cannot reach the API. Start the backend (`uvicorn app.main:app --reload` in backend/) and set VITE_API_BASE_URL.",
            { cause: err },
        );
    }
}

export default api;
