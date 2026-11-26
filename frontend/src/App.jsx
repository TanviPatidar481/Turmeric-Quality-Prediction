import { useCallback, useEffect, useRef, useState } from "react";
import { getModelCard, getSampleBlob, predictImage } from "./services/api";
import "./App.css";

const pct = (label) => (label ? label.split("_")[0] : "");

function Stat({ value, label, sub }) {
    return (
        <div className="stat">
            <div className="stat-value">{value}</div>
            <div className="stat-label">{label}</div>
            <div className="stat-sub">{sub}</div>
        </div>
    );
}

export default function App() {
    const [model, setModel] = useState(null);
    const [modelError, setModelError] = useState(null);
    const [samples, setSamples] = useState([]);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [fileName, setFileName] = useState(null);
    const [result, setResult] = useState(null);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState(null);
    const [dragging, setDragging] = useState(false);
    const [activeSample, setActiveSample] = useState(null);
    const inputRef = useRef(null);
    const urlRef = useRef(null);

    useEffect(() => {
        let alive = true;
        getModelCard()
            .then((m) => {
                if (!alive) return;
                setModel(m);
                setSamples(m.samples ?? []);
            })
            .catch((e) => alive && setModelError(e.message));
        return () => {
            alive = false;
        };
    }, []);

    useEffect(() => () => {
        if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    }, []);

    const showPreview = useCallback((blob) => {
        if (urlRef.current) URL.revokeObjectURL(urlRef.current);
        urlRef.current = URL.createObjectURL(blob);
        setPreviewUrl(urlRef.current);
    }, []);

    const run = useCallback(
        async (blob, name, sampleId = null) => {
            setBusy(true);
            setError(null);
            setResult(null);
            setActiveSample(sampleId);
            showPreview(blob);
            setFileName(name);
            try {
                setResult(await predictImage(blob));
            } catch (e) {
                setError(e.message);
            } finally {
                setBusy(false);
            }
        },
        [showPreview],
    );

    const onFiles = useCallback(
        (files) => {
            const file = files?.[0];
            if (!file) return;
            if (!file.type.startsWith("image/")) {
                setError("That file is not an image. Use a JPG or PNG.");
                return;
            }
            run(file, file.name);
        },
        [run],
    );

    const onSample = useCallback(
        async (sample) => {
            setBusy(true);
            setError(null);
            setResult(null);
            setActiveSample(sample.id);
            setFileName(`${sample.pretty} · sample`);
            try {
                const blob = await getSampleBlob(sample.id);
                showPreview(blob);
                setResult(await predictImage(blob));
            } catch (e) {
                setError(e.message);
            } finally {
                setBusy(false);
            }
        },
        [showPreview],
    );

    const sample = activeSample ? samples.find((s) => s.id === activeSample) : null;
    const agrees = sample && result ? sample.label === result.label : null;
    const ranked = result?.distances
        ? Object.entries(result.distances).sort((a, b) => a[1] - b[1])
        : [];
    const runnerUp = ranked[1];

    return (
        <div className="page">
            <header className="topbar">
                <div className="brand">
                    <span className="brand-dot" />
                    Turmeric&nbsp;Impurity
                </div>
                <div className="topbar-meta">
                    {model ? (
                        <span className="live">
                            <span className="live-dot" />
                            {model.classes.length} classes · LOSO validated
                        </span>
                    ) : modelError ? (
                        <span className="offline">backend offline</span>
                    ) : (
                        <span className="loading-dot">connecting…</span>
                    )}
                </div>
            </header>

            <section className="hero">
                <h1>
                    How much corn starch is <em>in this turmeric?</em>
                </h1>
                <p>
                    Drop a photo of the powder. The model white-balances against the paper in
                    the frame, masks out everything that isn’t powder, and reads the colour that
                    is left. Pure colour — no texture, no labels, no paper.
                </p>
            </section>

            {modelError && (
                <div className="banner error">
                    {modelError}
                    <span className="banner-hint">Start the API with <code>uvicorn app.main:app --port 8001</code>.</span>
                </div>
            )}

            <main className="grid">
                <section className="card">
                    <div className="card-head">
                        <h2>Photo</h2>
                        {result && <span className="tag">{fileName}</span>}
                    </div>

                    <div
                        className={`drop${dragging ? " dragging" : ""}${previewUrl ? " filled" : ""}`}
                        onDragOver={(e) => {
                            e.preventDefault();
                            setDragging(true);
                        }}
                        onDragLeave={() => setDragging(false)}
                        onDrop={(e) => {
                            e.preventDefault();
                            setDragging(false);
                            onFiles(e.dataTransfer.files);
                        }}
                        onClick={() => inputRef.current?.click()}
                        onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
                        }}
                        role="button"
                        tabIndex={0}
                        aria-label="Choose a turmeric photo"
                    >
                        {previewUrl ? (
                            <img src={previewUrl} alt="Selected turmeric spread" />
                        ) : (
                            <div className="drop-empty">
                                <div className="drop-icon">⌁</div>
                                <strong>Drop a photo, or click to choose</strong>
                                <span>JPG or PNG · keep the white paper visible</span>
                            </div>
                        )}
                        <input
                            ref={inputRef}
                            type="file"
                            accept="image/jpeg,image/png"
                            hidden
                            onChange={(e) => onFiles(e.target.files)}
                        />
                    </div>

                    <div className="samples">
                        <span className="samples-label">or try a real one</span>
                        <div className="chips">
                            {samples.map((s) => (
                                <button
                                    key={s.id}
                                    type="button"
                                    className={`chip${activeSample === s.id ? " on" : ""}`}
                                    onClick={() => onSample(s)}
                                    disabled={busy}
                                >
                                    {pct(s.label)}%
                                    <em>{s.note}</em>
                                </button>
                            ))}
                        </div>
                    </div>
                </section>

                <section className="card result">
                    <div className="card-head">
                        <h2>Verdict</h2>
                        {busy && <span className="tag">reading colour…</span>}
                    </div>

                    {error && <div className="banner error">{error}</div>}

                    {!result && !error && (
                        <div className="placeholder">
                            <p>
                                The model returns one of five levels. A photograph is reduced to
                                two numbers — the red and green share of the powder’s colour —
                                and matched to the closest class centroid.
                            </p>
                            {model && (
                                <div className="ladder idle">
                                    {model.classes.map((c) => (
                                        <div key={c} className="step">
                                            <span className="step-pct">{pct(c)}%</span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {result && (
                        <>
                            <div className="verdict">
                                <div className="verdict-num">
                                    {pct(result.label)}
                                    <span>%</span>
                                </div>
                                <div className="verdict-text">
                                    <strong>predicted corn-starch impurity</strong>
                                    <span>
                                        {runnerUp
                                            ? `next nearest level: ${pct(runnerUp[0])}%`
                                            : result.pretty}
                                    </span>
                                </div>
                            </div>

                            <div className="ladder">
                                {model?.classes.map((c) => (
                                    <div
                                        key={c}
                                        className={`step${c === result.label ? " on" : ""}`}
                                    >
                                        <span className="step-bar" />
                                        <span className="step-pct">{pct(c)}%</span>
                                    </div>
                                ))}
                            </div>

                            <div className="measured">
                                <div
                                    className="swatch"
                                    style={{ background: result.summary?.swatch ?? "#2b2b2b" }}
                                    title={`measured powder colour ${result.summary?.swatch ?? ""}`}
                                />
                                <dl>
                                    <div>
                                        <dt>r / (r+g+b)</dt>
                                        <dd>{result.features?.r_chroma?.toFixed(4) ?? "—"}</dd>
                                    </div>
                                    <div>
                                        <dt>g / (r+g+b)</dt>
                                        <dd>{result.features?.g_chroma?.toFixed(4) ?? "—"}</dd>
                                    </div>
                                    <div>
                                        <dt>powder in frame</dt>
                                        <dd>
                                            {result.summary
                                                ? `${Math.round(result.summary.powder_fraction * 100)}%`
                                                : "—"}
                                        </dd>
                                    </div>
                                </dl>
                            </div>

                            {result.summary?.preview && (
                                <figure className="region">
                                    <img
                                        src={result.summary.preview}
                                        alt="The powder region the model measured"
                                    />
                                    <figcaption>
                                        the region measured, after white balance
                                    </figcaption>
                                </figure>
                            )}

                            {sample && (
                                <p className={`sample-note${agrees ? "" : " miss"}`}>
                                    This sample is labelled {sample.pretty}.{" "}
                                    {agrees
                                        ? "The model agrees."
                                        : "The model disagrees — shown as-is."}
                                </p>
                            )}

                            {result.warnings?.length > 0 && (
                                <ul className="warnings">
                                    {result.warnings.map((w) => (
                                        <li key={w}>{w}</li>
                                    ))}
                                </ul>
                            )}
                        </>
                    )}
                </section>
            </main>

            {model && (
                <section className="card model">
                    <div className="card-head">
                        <h2>The model</h2>
                        <span className="mono">
                            {model.artifact} · {model.artifact_bytes} bytes ·{" "}
                            {model.features}
                        </span>
                    </div>

                    <div className="stats">
                        <Stat
                            value={`${(model.cv.photo_accuracy * 100).toFixed(1)}%`}
                            label="photos correct"
                            sub={`${model.n_photos} photos held out`}
                        />
                        <Stat
                            value={`${Math.round(model.cv.spread_accuracy * model.n_spreads)}/${model.n_spreads}`}
                            label="spreads correct"
                            sub="whole spreads held out"
                        />
                        <Stat
                            value={model.cv.photo_macro_f1.toFixed(3)}
                            label="macro F1"
                            sub="unweighted across classes"
                        />
                        <Stat
                            value={model.classes.length}
                            label="classes"
                            sub="30% folded into 40%"
                        />
                    </div>

                    <div className="model-cols">
                        <div>
                            <h3>Recall by class</h3>
                            <div className="recall">
                                {model.classes.map((c) => {
                                    const p = model.per_class[c];
                                    return (
                                        <div className="recall-row" key={c}>
                                            <span className="recall-name">{pct(c)}%</span>
                                            <span className="recall-bar">
                                                <i style={{ width: `${p.recall * 100}%` }} />
                                            </span>
                                            <span className="recall-val">
                                                {(p.recall * 100).toFixed(0)}%
                                            </span>
                                            <span className="recall-meta">
                                                <b>{p.n}</b> photos
                                                <br />
                                                {p.spreads} spreads
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        <div>
                            <h3>Confusion matrix</h3>
                            {model.confusion?.matrix?.length ? (
                                <>
                                    <table className="confusion">
                                        <thead>
                                            <tr>
                                                <th className="corner" />
                                                {model.confusion.labels.map((l) => (
                                                    <th key={l} scope="col">
                                                        {pct(l)}%
                                                    </th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {model.confusion.matrix.map((row, i) => {
                                                const actual = model.confusion.labels[i];
                                                return (
                                                    <tr key={actual}>
                                                        <th scope="row">{pct(actual)}%</th>
                                                        {row.map((v, j) => (
                                                            <td
                                                                key={j}
                                                                className={
                                                                    i === j
                                                                        ? "diag"
                                                                        : v > 0
                                                                          ? "off"
                                                                          : "zero"
                                                                }
                                                            >
                                                                {v}
                                                            </td>
                                                        ))}
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                    <p className="matrix-note">
                                        Rows are the true label, columns the prediction.
                                    </p>
                                </>
                            ) : (
                                <p className="matrix-note">Not stored in this artifact.</p>
                            )}
                        </div>
                    </div>

                    <div className="caveats">
                        <h3>What this can and cannot tell you</h3>
                        <ul>
                            <li>
                                <strong>30% and 40% are one class.</strong> The two batches are
                                only 0.0024 apart in colour against 0.008–0.013 for every other
                                pair, so the model reports them together as 40%.
                            </li>
                            <li>
                                <strong>0% vs 10% is the only fuzzy boundary.</strong> Every
                                error in the matrix sits there; 20%, 40% and 50% were never
                                confused.
                            </li>
                            <li>
                                <strong>Colour only.</strong> Texture, grind and cracks are
                                ignored on purpose — they track how the spread was made, not
                                how much starch is in it.
                            </li>
                            <li>
                                <strong>The paper must stay in the shot.</strong> The white
                                reference is what makes the colour comparable between photos.
                                A powder-only crop cannot be classified.
                            </li>
                            <li>
                                <strong>{model.n_spreads} independent spreads.</strong> Not 600:
                                the photos cluster into bursts of one spread each, and the
                                split above holds out whole spreads.
                            </li>
                        </ul>
                    </div>
                </section>
            )}

            <footer className="foot">
                <span>
                    nearest-centroid on paper-white-balanced powder chromaticity · colour only
                </span>
                <span className="mono">{model ? model.artifact : ""}</span>
            </footer>
        </div>
    );
}
