FROM node:20-bullseye AS base

# Enable corepack for pnpm/yarn if needed
RUN corepack enable

WORKDIR /app

############################
# 1) Install dependencies  #
############################

FROM base AS deps

RUN apt-get update && \
  apt-get install -y --no-install-recommends python3 python3-pip && \
  rm -rf /var/lib/apt/lists/*

# Install Node dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Install Python dependencies (legacy .docx parser + evaluation pipeline)
COPY scripts/requirements.txt ./scripts/requirements.txt
RUN python3 -m pip install --no-cache-dir -r scripts/requirements.txt

############################
# 2) Build the application #
############################

FROM deps AS build

COPY . .

ENV NODE_ENV=production

RUN npm run build

############################
# 3) Runtime image         #
############################

FROM node:20-bullseye AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000

# Install Python + parser dependency in the runtime image as well
RUN apt-get update && \
  apt-get install -y --no-install-recommends python3 python3-pip && \
  rm -rf /var/lib/apt/lists/*

COPY scripts/requirements.txt ./scripts/requirements.txt
RUN python3 -m pip install --no-cache-dir -r scripts/requirements.txt

# Copy node_modules and build output from build stage.
# package.json comes from deps (same file as npm ci used) so a bad build cache cannot leave an empty file here.
COPY --from=build /app/node_modules ./node_modules
COPY --from=deps /app/package.json ./package.json
COPY --from=build /app/next.config.ts ./next.config.ts
COPY --from=build /app/public ./public
COPY --from=build /app/.next ./.next
COPY --from=build /app/scripts ./scripts
COPY --from=build /app/evaluation_pipeline ./evaluation_pipeline

EXPOSE 3000

CMD ["npm", "start"]

