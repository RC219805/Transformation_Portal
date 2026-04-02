import { NextResponse } from "next/server.js";

export function proxy(request) {
  const response = NextResponse.next();
  response.headers.set("Referrer-Policy", "same-origin");
  response.headers.set("X-Frame-Options", "DENY");
  response.headers.set("X-Content-Type-Options", "nosniff");
  return response;
}

export const config = {
  matcher: ["/login"]
};
