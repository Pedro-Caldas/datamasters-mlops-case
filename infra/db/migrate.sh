#!/bin/sh
set -e

echo "Iniciando migração do Postgres..."

psql postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@postgres:5432/$POSTGRES_DB \
    -f /migrations/migrate.sql

echo "Migração concluída com sucesso!"
