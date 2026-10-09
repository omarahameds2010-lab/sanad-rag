# Build: docker build -t sanad-frontend ./frontend
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
ARG VITE_API_BASE_URL=http://localhost:8000
ARG VITE_SANAD_API_KEY=demo-key-123
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_SANAD_API_KEY=$VITE_SANAD_API_KEY
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
