// Backend origin for the frontend.
//
//   ""                              -> same origin (app served by the FastAPI
//                                      backend itself, e.g. on Render). Default.
//   "https://raglens.onrender.com"  -> absolute URL, for when the frontend is
//                                      hosted separately (a Hugging Face Static
//                                      Space, Cloudflare Pages, etc.).
//
// No trailing slash. The backend already sends permissive CORS headers.
window.RAGLENS_API_BASE = "";
