 

# Clasificando automáticamente lo que hoy hacemos a mano

*Cómo la tecnología puede ayudarnos a atender más rápido lo que ya sabemos hacer*

**Nota del expositor:** Presentarse brevemente y anunciar que esta charla busca mostrar, en lenguaje simple, un problema del día a día del negocio y una posible forma de aliviarlo con tecnología. Dejar claro que no se necesita conocimiento técnico para seguir la presentación.

---
# El problema

Imagina que cada día llegan **cientos de solicitudes**: peticiones, incidentes, preguntas, reclamos...

Cada una tiene:
- un **asunto** (un título corto),
- una **descripción** (el detalle de lo que necesita la persona),
- y algunos datos adicionales (tipo, idioma, prioridad, etc.).

```
📩 📩 📩 📩 📩 📩 📩 📩 ...
   ¿Quién las lee? ¿Quién decide qué hacer con cada una?
```

Alguien tiene que **leer una por una** y decidir a qué equipo o proceso debe ir.

**Nota del expositor:** Enfatizar que este es un problema muy común: cualquier área que reciba tickets, correos o solicitudes vive esta misma situación. Buscar que la audiencia se sienta identificada.

---

  ¿Cómo funciona hoy?

```
 Solicitud  --->  Persona revisa  --->  Clasificación  --->  Equipo responsable
   nueva          (lee y decide)         asignada              atiende
```

- La **intervención humana** ocurre en el paso central: alguien tiene que leer el contenido y decidir.
- Ese criterio muchas veces depende de la **experiencia** de esa persona.

**Nota del expositor:** Explicar que hoy el proceso funciona, pero depende completamente de que haya personas disponibles para leer y decidir cada caso, uno a uno.

---

  ¿Qué problemas genera?

| Situación | Efecto |
|---|---|
| 📈 Alto volumen de solicitudes | Se acumulan más rápido de lo que se pueden revisar |
| 🔁 Trabajo manual y repetitivo | Tiempo valioso dedicado a tareas rutinarias |
| 🧠 Depende del conocimiento de algunas personas | Si esa persona no está, el proceso se hace más lento |
| ❌ Posibles errores de clasificación | Una solicitud puede ir al equipo equivocado |
| 🔄 Reprocesos | Hay que reasignar y volver a revisar |
| ⏱️ Mayor tiempo de respuesta | La persona que solicitó algo espera más |
| 📉 Difícil de escalar | Más solicitudes = se necesitan más personas revisando |

**Nota del expositor:** No hace falta explicar todas, resaltar 2-3 que más resuenen con la audiencia (por ejemplo volumen y tiempo de respuesta).

---

# La oportunidad

> ¿Y si una máquina pudiera aprender de las decisiones que ya hemos tomado?

Hoy ya existe algo muy valioso: **historial de solicitudes ya clasificadas**.

```
Solicitudes pasadas + cómo fueron clasificadas = experiencia acumulada
```

Esa experiencia acumulada es exactamente lo que un modelo de **Machine Learning** puede aprovechar para aprender patrones.

**Nota del expositor:** Introducir la idea central de la charla: no partimos de cero, partimos de datos históricos que ya reflejan decisiones correctas tomadas por las personas.

---

  La solución propuesta

```
Solicitudes históricas  --->  Machine Learning  --->  Modelo entrenado
      (ya clasificadas)         (aprende patrones)

Nueva solicitud  --->  Modelo entrenado  --->  Clasificación sugerida  --->  Equipo responsable
```

La idea es simple: el modelo **aprende de lo que ya se hizo** y luego **sugiere** cómo clasificar cada solicitud nueva, de forma automática y casi inmediata.

**Nota del expositor:** Evitar mencionar algoritmos, código o matemáticas. El mensaje clave es "aprende de ejemplos pasados y sugiere una respuesta para casos nuevos".

---

  ¿Qué información utiliza el modelo?

El modelo mira la misma información que hoy revisa una persona:

- 📝 **Asunto** de la solicitud
- 📄 **Descripción** del caso
- 🏷️ **Tipo** de solicitud
- ⚙️ Otras variables disponibles (por ejemplo idioma o prioridad)

A partir de estos datos, el modelo busca **patrones** en los casos históricos ("solicitudes parecidas a esta se clasificaron así") y los usa para sugerir una categoría en los casos nuevos.

**Nota del expositor:** Reforzar que el modelo no "inventa" reglas: aprende de patrones que ya existían en las decisiones humanas pasadas.

---

  ¿Cómo sabemos si funciona?

Antes de usar el modelo en el día a día, lo ponemos a prueba con casos que **ya conocemos la respuesta correcta**.

> De 100 solicitudes que ya sabíamos cómo se clasificaron, ¿cuántas hubiera clasificado bien el modelo?

Algunas formas simples de medir esto:

- **Accuracy (precisión general):** de todas las solicitudes evaluadas, ¿qué porcentaje clasificó correctamente?
- **Precision:** cuando el modelo dice "esto es de tal equipo", ¿qué tan seguido acierta?
- **Recall:** de todos los casos que realmente eran de un equipo, ¿cuántos logró identificar el modelo?
- **F1-score:** un balance entre las dos anteriores, útil para tener una sola medida resumen.

**Nota del expositor:** No mostrar fórmulas. Usar la analogía de un examen con respuestas conocidas: se compara lo que dijo el modelo contra lo que realmente era correcto, caso por caso.

---

  ¿Qué ganamos?

- ⏳ Menor trabajo manual y repetitivo
- ⚡ Mayor velocidad de clasificación
- 🎯 Mayor consistencia (el modelo no se cansa ni se distrae)
- 📈 Escalabilidad: soporta más volumen sin necesitar más personas revisando
- 🤝 Apoyo a los equipos, no reemplazo de las personas
- 🔍 Permite priorizar la revisión humana en los casos más difíciles o dudosos

> El modelo es una **herramienta de apoyo a la decisión**, no un reemplazo de las personas.

**Nota del expositor:** Ser explícito en que la propuesta es de apoyo, no de sustitución. Esto suele reducir resistencia en la audiencia.

---

  Próximos pasos

```
Datos históricos  →  Preparación  →  Entrenamiento  →  Evaluación  →  Prueba  →  Implementación  →  Monitoreo
```

- Se espera validar el desempeño del modelo antes de usarlo en producción.
- El impacto real (tiempos, ahorro de esfuerzo) está **por evaluar** una vez se pruebe con casos reales.

> La tecnología no reemplaza el criterio humano, lo **potencia**: libera tiempo para que las personas se enfoquen en lo que realmente requiere su experiencia.

**Nota del expositor:** Cerrar conectando la propuesta técnica con el beneficio de negocio, dejando claro que los próximos pasos son incrementales y se validan en cada etapa antes de avanzar.
